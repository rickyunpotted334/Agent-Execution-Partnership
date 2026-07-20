from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from aep.actions.state_machine import validate_transition
from aep.audit.ledger import AuditLedger
from aep.contracts.models import (
    ActionRequest,
    ActionStatus,
    ExecutionEvidence,
    PolicyDecisionKind,
    TaskContract,
    VerificationResult,
)
from aep.policy.engine import PolicyEngine
from aep.recovery.service import RecoveryService
from aep.verification.service import VerificationService


class Adapter(Protocol):
    def execute(self, action: ActionRequest) -> dict[str, object]: ...


class ActionExecutionEngine:
    def __init__(self, adapters: dict[str, Adapter]) -> None:
        self.adapters = adapters
        self.policy = PolicyEngine()
        self.verifier = VerificationService()
        self.recovery = RecoveryService()
        self.audit = AuditLedger()
        self._states: dict[str, ActionStatus] = {}

    def _set_state(self, action_id: str, nxt: ActionStatus) -> None:
        current = self._states.get(action_id, ActionStatus.PROPOSED)
        if action_id not in self._states and nxt == ActionStatus.PROPOSED:
            self._states[action_id] = nxt
            return
        validate_transition(current, nxt)
        self._states[action_id] = nxt

    def execute(
        self, task: TaskContract, action: ActionRequest, correlation_id: str
    ) -> tuple[ExecutionEvidence, VerificationResult]:
        self._set_state(action.action_id, ActionStatus.VALIDATING)
        decision = self.policy.evaluate(task, action)

        if decision.decision == PolicyDecisionKind.DENY:
            self._set_state(action.action_id, ActionStatus.DENIED)
            evidence = self._denied_evidence(action, "policy_denied")
            result = self.verifier.verify(evidence, action.expected_effects)
            self.audit.append(
                actor=action.requested_by,
                task=task.task_id,
                action=action.action_id,
                event_type="action_denied",
                payload={"reason_codes": decision.reason_codes},
                correlation_id=correlation_id,
            )
            return evidence, result

        if decision.decision == PolicyDecisionKind.REQUIRE_APPROVAL:
            self._set_state(action.action_id, ActionStatus.AWAITING_APPROVAL)
            evidence = self._denied_evidence(action, "awaiting_approval")
            result = self.verifier.verify(evidence, action.expected_effects)
            return evidence, result

        self._set_state(action.action_id, ActionStatus.AUTHORIZED)
        self._set_state(action.action_id, ActionStatus.PRECONDITION_CHECK)
        self._set_state(action.action_id, ActionStatus.READY)
        self._set_state(action.action_id, ActionStatus.EXECUTING)

        adapter = self.adapters[action.channel]
        started_at = datetime.now(UTC)
        response = adapter.execute(action)
        completed_at = datetime.now(UTC)
        self._set_state(action.action_id, ActionStatus.VERIFYING)
        artifacts = response.get("artifacts", [])
        side_effects = response.get("side_effects", [])
        resource_usage_raw = response.get("resource_usage", {})
        resource_usage = cast(
            dict[str, Any], resource_usage_raw if isinstance(resource_usage_raw, dict) else {}
        )
        error_raw = response.get("error")
        error: str | None = str(error_raw) if error_raw is not None else None

        evidence = ExecutionEvidence(
            action_id=action.action_id,
            adapter=action.channel,
            started_at=started_at,
            completed_at=completed_at,
            status=str(response.get("status", "failed")),
            request_digest=_digest(action.model_dump()),
            response_digest=_digest(response),
            artifacts=[str(a) for a in artifacts] if isinstance(artifacts, list) else [],
            state_before_reference=f"obs:{action.observation_version}",
            state_after_reference=str(response.get("state_after", "unknown")),
            resource_usage=resource_usage,
            side_effects_detected=[str(s) for s in side_effects]
            if isinstance(side_effects, list)
            else [],
            logs_reference=str(response.get("logs_reference", "local://logs")),
            error=error,
            retryable=bool(response.get("retryable", False)),
            compensation_available=bool(response.get("compensation_available", False)),
        )

        verification = self.verifier.verify(evidence, action.expected_effects)
        if verification.verdict.value == "verified":
            self._set_state(action.action_id, ActionStatus.VERIFIED)
        else:
            self._set_state(action.action_id, ActionStatus.FAILED)
            self._set_state(action.action_id, ActionStatus.RECOVERING)
            _ = self.recovery.plan(action, evidence)

        self.audit.append(
            actor=action.requested_by,
            task=task.task_id,
            action=action.action_id,
            event_type="action_executed",
            payload={"status": evidence.status, "verification": verification.verdict.value},
            correlation_id=correlation_id,
        )
        return evidence, verification

    @staticmethod
    def _denied_evidence(action: ActionRequest, status: str) -> ExecutionEvidence:
        now = datetime.now(UTC)
        return ExecutionEvidence(
            action_id=action.action_id,
            adapter=action.channel,
            started_at=now,
            completed_at=now,
            status=status,
            request_digest=_digest(action.model_dump()),
            response_digest=_digest({"status": status}),
            artifacts=[],
            state_before_reference=f"obs:{action.observation_version}",
            state_after_reference=f"obs:{action.observation_version}",
            resource_usage={},
            side_effects_detected=[],
            logs_reference="local://policy",
            error=status,
            retryable=False,
            compensation_available=False,
        )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
