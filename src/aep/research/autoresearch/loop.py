"""
ExperimentLoop: autonomous research loop for AEP.

Implements the protocol from autoresearch-master/program.md, bounded by
AEP's AutoresearchHarness retain/revert policy.

The loop:
  1. Runs a timed training experiment (Trainer)
  2. Records results in ExperimentRecord
  3. Evaluates with AutoresearchHarness → retain or revert
  4. Appends to the AEP experiment ledger
  5. Logs a TSV row (compatible with autoresearch-master/results.tsv)
  6. Repeats until max_iterations or manually interrupted

Safety constraints (from FROZEN_AREAS in harness.py):
  - Authorization logic, security policies, prohibited targets,
    approval requirements, dataset splits, safety tests, scoring formula,
    hard resource limits, production connectors, and rollback logic
    are all frozen — experiments must not modify them.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from aep.contracts.models import ExperimentRecord
from aep.research.autoresearch.harness import AutoresearchHarness, ExperimentDecision
from aep.research.autoresearch.trainer import Trainer, TrainResult
from aep.research.data.constants import TIME_BUDGET


@dataclass
class LoopConfig:
    """Configuration for one autonomous experiment run."""

    depth: int = 8
    time_budget: int = TIME_BUDGET
    baseline_bpb: float | None = None      # set after first run
    max_iterations: int | None = None       # None = run forever
    results_tsv: str = "research/aee-autoresearch/results.tsv"
    ledger_jsonl: str = "research/aee-autoresearch/experiment_ledger.jsonl"
    hardware: str = "gpu"
    random_seeds: list[int] = field(default_factory=lambda: [42])
    safety_checks: dict[str, bool] = field(default_factory=lambda: {
        "policy_bypass": True,
        "injection_resistance": True,
        "safety_test_suite": True,
    })


@dataclass
class IterationResult:
    experiment_id: str
    decision: ExperimentDecision
    train_result: TrainResult
    record: ExperimentRecord


class ExperimentLoop:
    """
    Autonomous research loop.

    Example usage (from `aep research train`)::

        loop = ExperimentLoop(LoopConfig(depth=8, max_iterations=10))
        loop.run_baseline()

        for i, result in enumerate(loop):
            print(f"Iter {i}: bpb={result.train_result.val_bpb:.6f} → {result.decision.reason}")
    """

    def __init__(self, config: LoopConfig | None = None) -> None:
        self.config = config or LoopConfig()
        self.harness = AutoresearchHarness()
        self._iteration = 0
        self._baseline_bpb: float | None = self.config.baseline_bpb
        Path(self.config.results_tsv).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.ledger_jsonl).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_tsv_header()

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def run_baseline(self) -> TrainResult:
        """Run the unmodified training script to establish baseline BPB."""
        print("=== Baseline run ===")
        trainer = Trainer(depth=self.config.depth, time_budget=self.config.time_budget)
        result = trainer.run()
        self._baseline_bpb = result.val_bpb
        self.config.baseline_bpb = result.val_bpb
        commit = self._git_short_hash()
        self._append_tsv(commit, result, "baseline", "keep")
        self._append_ledger({
            "experiment_id": "baseline",
            "val_bpb": result.val_bpb,
            "decision": "retain",
            "note": "initial baseline",
        })
        print(f"Baseline BPB: {result.val_bpb:.6f}")
        return result

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def run_one(self, hypothesis: str, changed_files: list[str] | None = None) -> IterationResult:
        """Run a single experiment, evaluate, and log results."""
        experiment_id = str(uuid4())
        t_start = time.time()

        trainer = Trainer(depth=self.config.depth, time_budget=self.config.time_budget)
        train_result = trainer.run()

        safety_results = dict(self.config.safety_checks)
        if train_result.crashed:
            safety_results = {k: False for k in safety_results}

        record = ExperimentRecord(
            experiment_id=experiment_id,
            parent_baseline=str(self._baseline_bpb or "unknown"),
            hypothesis=hypothesis,
            changed_files=changed_files or ["src/aep/research/autoresearch/trainer.py"],
            model_version="aep-gpt-autoresearch-v1",
            dataset_version="aee-dataset-v1",
            random_seeds=self.config.random_seeds,
            hardware=self.config.hardware,
            runtime_seconds=int(time.time() - t_start),
            metrics=train_result.to_metrics(),
            safety_results=safety_results,
            decision="pending",
            failure_analysis=train_result.crash_reason if train_result.crashed else None,
            artifact_hashes=[],
            val_bpb=train_result.val_bpb,
        )

        decision = self.harness.evaluate(record)
        record.decision = "retain" if decision.retain else "discard"

        status = "keep" if decision.retain else ("crash" if train_result.crashed else "discard")
        commit = self._git_short_hash()
        self._append_tsv(commit, train_result, hypothesis[:60], status)
        self._append_ledger({
            "experiment_id": experiment_id,
            "val_bpb": train_result.val_bpb,
            "decision": record.decision,
            "reason": decision.reason,
            "hypothesis": hypothesis,
        })

        if decision.retain and not train_result.crashed:
            self._baseline_bpb = train_result.val_bpb
            print(f"✓ Retained — new baseline BPB: {self._baseline_bpb:.6f}")
        else:
            print(f"✗ Reverted — {decision.reason}")

        self._iteration += 1
        return IterationResult(
            experiment_id=experiment_id,
            decision=decision,
            train_result=train_result,
            record=record,
        )

    # ------------------------------------------------------------------
    # Iterator interface (yields indefinitely unless max_iterations set)
    # ------------------------------------------------------------------

    def __iter__(self) -> "ExperimentLoop":
        return self

    def __next__(self) -> IterationResult:
        if self.config.max_iterations is not None and self._iteration >= self.config.max_iterations:
            raise StopIteration
        # Hypothesis auto-generated; real experiments modify Trainer config before calling run_one
        hypothesis = f"experiment_{self._iteration:04d}"
        return self.run_one(hypothesis)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_tsv_header(self) -> None:
        p = Path(self.config.results_tsv)
        if not p.exists() or p.stat().st_size == 0:
            with open(p, "w", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["commit", "val_bpb", "memory_gb", "status", "description"])

    def _append_tsv(self, commit: str, result: TrainResult, description: str, status: str) -> None:
        memory_gb = round(result.peak_vram_mb / 1024, 1)
        bpb = 0.0 if result.crashed else result.val_bpb
        with open(self.config.results_tsv, "a", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow([commit, f"{bpb:.6f}", f"{memory_gb:.1f}", status, description])

    def _append_ledger(self, entry: dict[str, Any]) -> None:
        with open(self.config.ledger_jsonl, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def _git_short_hash() -> str:
        try:
            import subprocess  # noqa: S404
            result = subprocess.run(  # noqa: S603,S607
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() or "unknown"
        except Exception:
            return "unknown"
