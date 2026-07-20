"""
Unit tests for autoresearch integration from autoresearch-master.

Covers:
  - GPTConfig  (no torch required)
  - data.constants
  - data.pipeline (non-GPU surface)
  - AutoresearchHarness BPB metric logic
  - TrainResult.to_metrics()
  - ExperimentLoop helpers (TSV/ledger/git)
  - LoopConfig defaults
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from aep.contracts.models import ExperimentRecord
from aep.models.gpt.config import GPTConfig
from aep.research.autoresearch.harness import AutoresearchHarness
from aep.research.autoresearch.loop import ExperimentLoop, LoopConfig
from aep.research.autoresearch.trainer import TrainResult
from aep.research.data.constants import (
    MAX_SEQ_LEN,
    TIME_BUDGET,
    VOCAB_SIZE,
    WINDOW_PATTERN,
)
from aep.research.data.pipeline import PREPARE_DEPS_AVAILABLE


# ---------------------------------------------------------------------------
# GPTConfig
# ---------------------------------------------------------------------------

class TestGPTConfig:
    def test_small_config(self) -> None:
        cfg = GPTConfig.small()
        assert cfg.sequence_len == 64
        assert cfg.vocab_size == 256
        assert cfg.n_layer == 2
        assert cfg.n_embd == 64

    def test_from_depth(self) -> None:
        cfg = GPTConfig.from_depth(depth=8, vocab_size=8192)
        assert cfg.n_layer == 8
        assert cfg.vocab_size == 8192
        # model_dim must be divisible by head_dim
        assert cfg.n_embd % 128 == 0
        assert cfg.n_head == cfg.n_embd // 128

    def test_to_dict(self) -> None:
        cfg = GPTConfig.small()
        d = cfg.to_dict()
        assert d["sequence_len"] == 64
        assert d["vocab_size"] == 256

    def test_window_pattern_default(self) -> None:
        cfg = GPTConfig()
        assert cfg.window_pattern == "SSSL"

    def test_aspect_ratio_scaling(self) -> None:
        cfg8 = GPTConfig.from_depth(depth=8, vocab_size=4096)
        cfg12 = GPTConfig.from_depth(depth=12, vocab_size=4096)
        # Deeper model should have more params
        assert cfg12.n_embd >= cfg8.n_embd


# ---------------------------------------------------------------------------
# data.constants
# ---------------------------------------------------------------------------

class TestDataConstants:
    def test_time_budget(self) -> None:
        assert TIME_BUDGET == 300  # must not change — fixed eval budget

    def test_max_seq_len(self) -> None:
        assert MAX_SEQ_LEN == 2048

    def test_vocab_size(self) -> None:
        assert VOCAB_SIZE == 8192

    def test_window_pattern(self) -> None:
        assert all(c in "SL" for c in WINDOW_PATTERN)


# ---------------------------------------------------------------------------
# data.pipeline — non-GPU surface
# ---------------------------------------------------------------------------

class TestPipelineSurface:
    def test_prepare_deps_flag_is_bool(self) -> None:
        assert isinstance(PREPARE_DEPS_AVAILABLE, bool)

    def test_tokenizer_requires_deps_when_missing(self) -> None:
        if PREPARE_DEPS_AVAILABLE:
            pytest.skip("deps available — skip stub test")
        from aep.research.data.pipeline import Tokenizer
        with pytest.raises(RuntimeError, match="tiktoken"):
            Tokenizer.from_directory()

    def test_download_data_requires_deps_when_missing(self) -> None:
        if PREPARE_DEPS_AVAILABLE:
            pytest.skip("deps available — skip stub test")
        from aep.research.data.pipeline import download_data
        with pytest.raises(RuntimeError):
            download_data(num_shards=1)


# ---------------------------------------------------------------------------
# AutoresearchHarness — BPB metric paths
# ---------------------------------------------------------------------------

def _make_record(**overrides: object) -> ExperimentRecord:
    defaults: dict[str, object] = dict(
        parent_baseline="1.500000",
        hypothesis="test",
        changed_files=["train.py"],
        model_version="v1",
        dataset_version="aee-dataset-v1",
        random_seeds=[42],
        hardware="gpu",
        runtime_seconds=300,
        metrics={"end_to_end_completion_rate": 0.3},
        safety_results={"policy_bypass": True, "injection_resistance": True},
        decision="pending",
        val_bpb=1.45,
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)  # type: ignore[arg-type]


class TestHarnessBPB:
    def test_bpb_improvement_retained(self) -> None:
        h = AutoresearchHarness()
        d = h.evaluate(_make_record(val_bpb=1.490))  # 0.010 improvement
        assert d.retain
        assert "bpb_improved" in d.reason

    def test_bpb_large_improvement_retained(self) -> None:
        h = AutoresearchHarness()
        d = h.evaluate(_make_record(val_bpb=1.200))
        assert d.retain

    def test_bpb_regression_reverted(self) -> None:
        h = AutoresearchHarness()
        d = h.evaluate(_make_record(val_bpb=1.510))
        assert not d.retain
        assert "bpb_regressed" in d.reason

    def test_bpb_trivial_improvement_reverted(self) -> None:
        # improvement < 0.001 → not worth the complexity
        h = AutoresearchHarness()
        d = h.evaluate(_make_record(val_bpb=1.4995))
        assert not d.retain

    def test_safety_failure_always_reverts(self) -> None:
        h = AutoresearchHarness()
        d = h.evaluate(_make_record(
            val_bpb=0.500,  # great BPB, but safety failed
            safety_results={"policy_bypass": False},
        ))
        assert not d.retain
        assert "safety" in d.reason

    def test_crash_reverted(self) -> None:
        h = AutoresearchHarness()
        d = h.evaluate(_make_record(
            val_bpb=0.0,
            failure_analysis="OOM at step 5",
        ))
        assert not d.retain
        assert "crash" in d.reason

    def test_non_numeric_baseline_falls_through_to_completion_rate(self) -> None:
        h = AutoresearchHarness()
        d = h.evaluate(_make_record(
            parent_baseline="unknown",
            val_bpb=1.200,
            metrics={"end_to_end_completion_rate": 0.8},
        ))
        assert d.retain

    def test_no_val_bpb_uses_completion_rate(self) -> None:
        h = AutoresearchHarness()
        d = h.evaluate(_make_record(
            parent_baseline="b1",  # non-numeric → falls through
            val_bpb=None,
            metrics={"end_to_end_completion_rate": 0.9},
        ))
        assert d.retain


# ---------------------------------------------------------------------------
# TrainResult.to_metrics()
# ---------------------------------------------------------------------------

class TestTrainResult:
    def test_to_metrics_shape(self) -> None:
        r = TrainResult(
            val_bpb=1.234,
            training_seconds=295.0,
            total_seconds=320.0,
            peak_vram_mb=44000.0,
            mfu_percent=38.5,
            total_tokens_m=490.0,
            num_steps=940,
            num_params_m=50.3,
            depth=8,
        )
        m = r.to_metrics()
        assert m["val_bpb"] == pytest.approx(1.234)
        assert "end_to_end_completion_rate" in m
        assert 0.0 <= m["end_to_end_completion_rate"] <= 1.0

    def test_crashed_result_metrics(self) -> None:
        r = TrainResult(
            val_bpb=0.0,
            training_seconds=0,
            total_seconds=5,
            peak_vram_mb=0,
            mfu_percent=0,
            total_tokens_m=0,
            num_steps=0,
            num_params_m=50.3,
            depth=8,
            crashed=True,
            crash_reason="OOM",
        )
        assert r.crashed
        assert r.crash_reason == "OOM"


# ---------------------------------------------------------------------------
# ExperimentLoop helpers (file I/O, no GPU needed)
# ---------------------------------------------------------------------------

class TestExperimentLoopHelpers:
    def test_tsv_header_created(self, tmp_path: Path) -> None:
        tsv = str(tmp_path / "results.tsv")
        ledger = str(tmp_path / "ledger.jsonl")
        loop = ExperimentLoop(LoopConfig(
            results_tsv=tsv,
            ledger_jsonl=ledger,
            max_iterations=0,
        ))
        assert Path(tsv).exists()
        header = Path(tsv).read_text()
        assert "val_bpb" in header
        assert "commit" in header

    def test_append_ledger(self, tmp_path: Path) -> None:
        tsv = str(tmp_path / "results.tsv")
        ledger = str(tmp_path / "ledger.jsonl")
        loop = ExperimentLoop(LoopConfig(results_tsv=tsv, ledger_jsonl=ledger, max_iterations=0))
        loop._append_ledger({"experiment_id": "x", "val_bpb": 1.23, "decision": "retain"})
        lines = Path(ledger).read_text().splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["val_bpb"] == pytest.approx(1.23)

    def test_append_tsv_row(self, tmp_path: Path) -> None:
        tsv = str(tmp_path / "results.tsv")
        ledger = str(tmp_path / "ledger.jsonl")
        loop = ExperimentLoop(LoopConfig(results_tsv=tsv, ledger_jsonl=ledger, max_iterations=0))
        result = TrainResult(
            val_bpb=0.987,
            training_seconds=300,
            total_seconds=325,
            peak_vram_mb=44032,
            mfu_percent=39.8,
            total_tokens_m=499,
            num_steps=950,
            num_params_m=50.3,
            depth=8,
        )
        loop._append_tsv("abc1234", result, "test hypothesis", "keep")
        rows = Path(tsv).read_text().splitlines()
        assert len(rows) == 2  # header + 1 data row
        assert "abc1234" in rows[1]
        assert "0.987000" in rows[1]

    def test_git_short_hash_returns_string(self) -> None:
        h = ExperimentLoop._git_short_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_loop_config_defaults(self) -> None:
        cfg = LoopConfig()
        assert cfg.depth == 8
        assert cfg.time_budget == TIME_BUDGET
        assert cfg.max_iterations is None
        assert "policy_bypass" in cfg.safety_checks

    def test_max_iterations_stops_loop(self, tmp_path: Path) -> None:
        """Loop with max_iterations=0 should immediately raise StopIteration."""
        tsv = str(tmp_path / "results.tsv")
        ledger = str(tmp_path / "ledger.jsonl")
        loop = ExperimentLoop(LoopConfig(results_tsv=tsv, ledger_jsonl=ledger, max_iterations=0))
        results = list(loop)
        assert results == []
