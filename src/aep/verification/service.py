from __future__ import annotations

from aep.contracts.models import ExecutionEvidence, VerificationResult, VerificationVerdict


class VerificationService:
    def verify(
        self, evidence: ExecutionEvidence, expected_effects: list[str]
    ) -> VerificationResult:
        ok = evidence.status == "success" and evidence.error is None
        return VerificationResult(
            action_id=evidence.action_id,
            expected_effects_verified=ok,
            prohibited_effects_absent=not evidence.side_effects_detected,
            completion_criteria_progress=1.0 if ok else 0.0,
            confidence=0.9 if ok else 0.3,
            evidence_references=[evidence.execution_id, evidence.logs_reference],
            verdict=VerificationVerdict.VERIFIED if ok else VerificationVerdict.FAILED,
            recommended_next_state="continue" if ok else "recover",
        )
