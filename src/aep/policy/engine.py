from __future__ import annotations

from datetime import UTC, datetime, timedelta

from aep.config.settings import get_settings
from aep.contracts.models import (
    ActionRequest,
    PolicyDecision,
    PolicyDecisionKind,
    RiskClass,
    TaskContract,
)


class PolicyEngine:
    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate(self, task: TaskContract, action: ActionRequest) -> PolicyDecision:
        reasons: list[str] = []
        required_approvals: list[str] = []
        decision = PolicyDecisionKind.ALLOW

        if action.operation not in task.allowed_tools and task.allowed_tools:
            return PolicyDecision(
                action_id=action.action_id,
                decision=PolicyDecisionKind.DENY,
                reason_codes=["tool_not_allowed"],
                required_approvals=[],
                constraints_added=[],
                expiration=datetime.now(UTC) + timedelta(minutes=5),
                policy_version=self.settings.policy_version,
            )

        if action.risk_class in {
            RiskClass.CONSEQUENTIAL_WRITE,
            RiskClass.FINANCIAL,
            RiskClass.PRIVILEGED_ADMINISTRATIVE,
            RiskClass.PHYSICAL_OR_SAFETY_RELATED,
        }:
            decision = PolicyDecisionKind.REQUIRE_APPROVAL
            required_approvals.append("human_principal")
            reasons.append("high_risk_action")

        if action.risk_class == RiskClass.FINANCIAL:
            reasons.append("financial_limit_enforced")
            required_approvals.append("finance_authorizer")

        if task.risk_budget <= 0:
            decision = PolicyDecisionKind.DENY
            reasons.append("risk_budget_exhausted")

        return PolicyDecision(
            action_id=action.action_id,
            decision=decision,
            reason_codes=reasons,
            required_approvals=required_approvals,
            constraints_added=[],
            expiration=datetime.now(UTC) + timedelta(minutes=5),
            policy_version=self.settings.policy_version,
        )
