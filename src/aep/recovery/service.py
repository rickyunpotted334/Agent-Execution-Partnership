from __future__ import annotations

from dataclasses import dataclass

from aep.contracts.models import ActionRequest, ExecutionEvidence


@dataclass
class RecoveryPlan:
    action_id: str
    strategy: str
    retry_allowed: bool
    compensation_available: bool


class RecoveryService:
    def plan(self, action: ActionRequest, evidence: ExecutionEvidence) -> RecoveryPlan:
        retry_allowed = (
            evidence.retryable
            and action.risk_class.name in {"READ_ONLY", "REVERSIBLE_WRITE"}
            and action.idempotency_key != ""
        )
        return RecoveryPlan(
            action_id=action.action_id,
            strategy="retry" if retry_allowed else "escalate",
            retry_allowed=retry_allowed,
            compensation_available=evidence.compensation_available,
        )
