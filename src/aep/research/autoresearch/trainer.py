"""
Trainer: orchestrates a single timed training run.

Wraps the model, optimizer, dataloader, and schedule from
autoresearch-master/train.py into an AEP-bounded component that:
  - Runs for a fixed TIME_BUDGET (default 300 s)
  - Returns an ExperimentMetrics record that feeds into AutoresearchHarness
  - Aborts cleanly on NaN/exploding loss

Requires torch + CUDA + tokenizer data (run `aep research prepare` first).
"""
from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field
from typing import Any

from aep.research.data.constants import (
    ADAM_BETAS,
    DEFAULT_DEPTH,
    DEVICE_BATCH_SIZE,
    EMBEDDING_LR,
    FINAL_LR_FRAC,
    HEAD_DIM,
    MATRIX_LR,
    MAX_SEQ_LEN,
    SCALAR_LR,
    TIME_BUDGET,
    TOTAL_BATCH_SIZE,
    UNEMBEDDING_LR,
    WARMDOWN_RATIO,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    WINDOW_PATTERN,
)


@dataclass
class TrainResult:
    """Results from a single training run, mapped to AEP ExperimentRecord metrics."""

    val_bpb: float
    training_seconds: float
    total_seconds: float
    peak_vram_mb: float
    mfu_percent: float
    total_tokens_m: float
    num_steps: int
    num_params_m: float
    depth: int
    crashed: bool = False
    crash_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metrics(self) -> dict[str, float]:
        """Convert to the dict format expected by ExperimentRecord.metrics."""
        return {
            "val_bpb": self.val_bpb,
            "training_seconds": self.training_seconds,
            "total_seconds": self.total_seconds,
            "peak_vram_mb": self.peak_vram_mb,
            "mfu_percent": self.mfu_percent,
            "total_tokens_m": self.total_tokens_m,
            "num_steps": float(self.num_steps),
            "num_params_m": self.num_params_m,
            # AEP completion-rate proxy: normalised BPB improvement over random baseline
            "end_to_end_completion_rate": max(0.0, 1.0 - self.val_bpb / 4.0),
        }


class Trainer:
    """
    Single-GPU training harness.

    Usage::

        trainer = Trainer(depth=8)
        result  = trainer.run()
    """

    # H100 BF16 TFLOPS (used for MFU estimate only)
    H100_BF16_PEAK_FLOPS: float = 989.5e12

    def __init__(
        self,
        depth: int = DEFAULT_DEPTH,
        time_budget: int = TIME_BUDGET,
        device_batch_size: int = DEVICE_BATCH_SIZE,
        total_batch_size: int = TOTAL_BATCH_SIZE,
        matrix_lr: float = MATRIX_LR,
        embedding_lr: float = EMBEDDING_LR,
        unembedding_lr: float = UNEMBEDDING_LR,
        scalar_lr: float = SCALAR_LR,
        weight_decay: float = WEIGHT_DECAY,
        adam_betas: tuple[float, float] = ADAM_BETAS,
        warmup_ratio: float = WARMUP_RATIO,
        warmdown_ratio: float = WARMDOWN_RATIO,
        final_lr_frac: float = FINAL_LR_FRAC,
        window_pattern: str = WINDOW_PATTERN,
    ) -> None:
        self.depth = depth
        self.time_budget = time_budget
        self.device_batch_size = device_batch_size
        self.total_batch_size = total_batch_size
        self.matrix_lr = matrix_lr
        self.embedding_lr = embedding_lr
        self.unembedding_lr = unembedding_lr
        self.scalar_lr = scalar_lr
        self.weight_decay = weight_decay
        self.adam_betas = adam_betas
        self.warmup_ratio = warmup_ratio
        self.warmdown_ratio = warmdown_ratio
        self.final_lr_frac = final_lr_frac
        self.window_pattern = window_pattern

    # ------------------------------------------------------------------
    # LR / momentum schedules (identical to train.py)
    # ------------------------------------------------------------------

    def _lr_multiplier(self, progress: float) -> float:
        if progress < self.warmup_ratio:
            return progress / self.warmup_ratio if self.warmup_ratio > 0 else 1.0
        if progress < 1.0 - self.warmdown_ratio:
            return 1.0
        cooldown = (1.0 - progress) / self.warmdown_ratio
        return cooldown + (1 - cooldown) * self.final_lr_frac

    @staticmethod
    def _muon_momentum(step: int) -> float:
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def _weight_decay(self, progress: float) -> float:
        return self.weight_decay * (1 - progress)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> TrainResult:
        try:
            import torch
        except ImportError as e:
            return TrainResult(
                val_bpb=0.0, training_seconds=0, total_seconds=0,
                peak_vram_mb=0, mfu_percent=0, total_tokens_m=0,
                num_steps=0, num_params_m=0, depth=self.depth,
                crashed=True, crash_reason=f"torch not installed: {e}",
            )

        from aep.models.gpt.config import GPTConfig
        from aep.models.gpt.model import GPT
        from aep.research.data.pipeline import (
            Tokenizer, make_dataloader, evaluate_bpb, PREPARE_DEPS_AVAILABLE
        )

        if not PREPARE_DEPS_AVAILABLE:
            return TrainResult(
                val_bpb=0.0, training_seconds=0, total_seconds=0,
                peak_vram_mb=0, mfu_percent=0, total_tokens_m=0,
                num_steps=0, num_params_m=0, depth=self.depth,
                crashed=True, crash_reason="prepare deps not installed",
            )

        t_start = time.time()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        tokenizer = Tokenizer.from_directory()
        vocab_size = tokenizer.get_vocab_size()

        config = GPTConfig.from_depth(self.depth, vocab_size)
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()

        num_flops_per_token = model.estimate_flops()
        num_params = model.num_scaling_params()["total"]

        tokens_per_step = self.device_batch_size * MAX_SEQ_LEN
        assert self.total_batch_size % tokens_per_step == 0
        grad_accum_steps = self.total_batch_size // tokens_per_step

        optimizer = model.setup_optimizer(
            unembedding_lr=self.unembedding_lr,
            embedding_lr=self.embedding_lr,
            matrix_lr=self.matrix_lr,
            weight_decay=self.weight_decay,
            adam_betas=self.adam_betas,
            scalar_lr=self.scalar_lr,
        )

        try:
            model = torch.compile(model, dynamic=False)
        except Exception:
            pass  # eager fallback

        train_loader = make_dataloader(tokenizer, self.device_batch_size, MAX_SEQ_LEN, "train")
        x, y, epoch = next(train_loader)

        smooth_train_loss = 0.0
        total_training_time = 0.0
        step = 0

        while True:
            torch.cuda.synchronize()
            t0 = time.time()

            for _ in range(grad_accum_steps):
                with autocast_ctx:
                    loss = model(x, y)
                train_loss = loss.detach()
                (loss / grad_accum_steps).backward()
                x, y, epoch = next(train_loader)

            progress = min(total_training_time / self.time_budget, 1.0)
            lrm = self._lr_multiplier(progress)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group["kind"] == "muon":
                    group["momentum"] = self._muon_momentum(step)
                    group["weight_decay"] = self._weight_decay(progress)
            optimizer.step()
            model.zero_grad(set_to_none=True)

            loss_f = train_loss.item()
            if math.isnan(loss_f) or loss_f > 100:
                return TrainResult(
                    val_bpb=0.0, training_seconds=total_training_time,
                    total_seconds=time.time() - t_start, peak_vram_mb=0,
                    mfu_percent=0, total_tokens_m=0, num_steps=step,
                    num_params_m=num_params / 1e6, depth=self.depth,
                    crashed=True, crash_reason=f"loss exploded at step {step}: {loss_f}",
                )

            torch.cuda.synchronize()
            dt = time.time() - t0
            if step > 10:
                total_training_time += dt

            ema_beta = 0.9
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * loss_f
            debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
            tok_per_sec = int(self.total_batch_size / dt)
            mfu = 100 * num_flops_per_token * self.total_batch_size / dt / self.H100_BF16_PEAK_FLOPS
            remaining = max(0, self.time_budget - total_training_time)

            print(
                f"\rstep {step:05d} ({100*progress:.1f}%) | "
                f"loss: {debiased:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | "
                f"tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | remaining: {remaining:.0f}s  ",
                end="", flush=True,
            )

            if step == 0:
                gc.collect(); gc.freeze(); gc.disable()
            elif (step + 1) % 5000 == 0:
                gc.collect()

            step += 1
            if total_training_time >= self.time_budget:
                break

        print()  # newline after training progress
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        val_bpb = evaluate_bpb(model, tokenizer, self.device_batch_size)
        total_seconds = time.time() - t_start

        summary = (
            f"---\n"
            f"val_bpb:          {val_bpb:.6f}\n"
            f"training_seconds: {total_training_time:.1f}\n"
            f"total_seconds:    {total_seconds:.1f}\n"
            f"peak_vram_mb:     {peak_vram_mb:.1f}\n"
            f"mfu_percent:      {100*num_flops_per_token*self.total_batch_size/dt/self.H100_BF16_PEAK_FLOPS:.2f}\n"
            f"total_tokens_M:   {step * self.total_batch_size / 1e6:.1f}\n"
            f"num_steps:        {step}\n"
            f"num_params_M:     {num_params/1e6:.1f}\n"
            f"depth:            {self.depth}\n"
        )
        print(summary)

        return TrainResult(
            val_bpb=val_bpb,
            training_seconds=total_training_time,
            total_seconds=total_seconds,
            peak_vram_mb=peak_vram_mb,
            mfu_percent=100 * num_flops_per_token * self.total_batch_size / dt / self.H100_BF16_PEAK_FLOPS,
            total_tokens_m=step * self.total_batch_size / 1e6,
            num_steps=step,
            num_params_m=num_params / 1e6,
            depth=self.depth,
        )
