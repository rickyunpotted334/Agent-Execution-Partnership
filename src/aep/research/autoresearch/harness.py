from __future__ import annotations

from dataclasses import dataclass

from aep.contracts.models import ExperimentRecord

FROZEN_AREAS = {
    "authorization_logic",
    "security_policies",
    "prohibited_targets",
    "approval_requirements",
    "dataset_splits",
    "safety_tests",
    "scoring_formula",
    "hard_resource_limits",
    "production_connectors",
    "rollback_logic",
}

ALLOWED_AREAS = {
    "training_hyperparameters",
    "fine_tuning_config",
    "tool_serialization",
    "observation_summarization",
    "action_selection_prompts",
    "recovery_heuristics",
    "verifier_thresholds",
    "locator_heuristics",
}

# Minimum BPB improvement required to retain a change (simplicity criterion:
# a trivial improvement is not worth added complexity)
_MIN_BPB_IMPROVEMENT: float = 0.001


@dataclass
class ExperimentDecision:
    retain: bool
    reason: str


class AutoresearchHarness:
    """Evaluate a training experiment and decide whether to retain or revert.

    Decision logic (in priority order):
    1. Safety failure → always revert.
    2. Crashed (val_bpb == 0) → revert.
    3. val_bpb improved by at least _MIN_BPB_IMPROVEMENT over the parent baseline → retain.
    4. end_to_end_completion_rate > 0.5 (fallback for non-BPB experiments) → retain.
    5. Otherwise → revert.
    """

    def evaluate(self, record: ExperimentRecord) -> ExperimentDecision:
        safety_ok = all(record.safety_results.values())
        if not safety_ok:
            return ExperimentDecision(retain=False, reason="safety_failed_auto_revert")

        # Crash check
        val_bpb = record.val_bpb if record.val_bpb is not None else record.metrics.get("val_bpb", 0.0)
        if val_bpb == 0.0 and record.failure_analysis is not None:
            return ExperimentDecision(retain=False, reason="crash_auto_revert")

        # BPB improvement check
        try:
            baseline_bpb = float(record.parent_baseline)
            if baseline_bpb > 0 and val_bpb > 0:
                improvement = baseline_bpb - val_bpb
                if improvement >= _MIN_BPB_IMPROVEMENT:
                    return ExperimentDecision(
                        retain=True,
                        reason=f"bpb_improved_by_{improvement:.6f}",
                    )
                if improvement < 0:
                    return ExperimentDecision(
                        retain=False,
                        reason=f"bpb_regressed_by_{-improvement:.6f}",
                    )
        except (ValueError, TypeError):
            pass  # parent_baseline not numeric — fall through

        # Fallback: legacy completion rate
        improved = record.metrics.get("end_to_end_completion_rate", 0.0) > 0.5
        if improved:
            return ExperimentDecision(retain=True, reason="acceptance_criteria_met")
        return ExperimentDecision(retain=False, reason="no_meaningful_improvement")
