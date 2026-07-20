from aep.contracts.models import ActionRequest, PolicyDecisionKind, RiskClass, TaskContract
from aep.policy.engine import PolicyEngine


def _task() -> TaskContract:
    return TaskContract(
        goal="test",
        human_principal="alice",
        agent_identity="agent.local",
        completion_criteria=["done"],
        idempotency_key="k1",
        allowed_tools=["list_dir"],
    )


def test_policy_denies_unallowed_tool() -> None:
    decision = PolicyEngine().evaluate(
        _task(),
        ActionRequest(
            task_id="t1",
            operation="delete_file",
            channel="filesystem",
            target={"path": "x"},
            risk_class=RiskClass.REVERSIBLE_WRITE,
            idempotency_key="a1",
            requested_by="agent.local",
            observation_version="v1",
        ),
    )
    assert decision.decision == PolicyDecisionKind.DENY


def test_financial_requires_approval() -> None:
    decision = PolicyEngine().evaluate(
        _task(),
        ActionRequest(
            task_id="t1",
            operation="list_dir",
            channel="filesystem",
            target={"path": "x"},
            risk_class=RiskClass.FINANCIAL,
            idempotency_key="a2",
            requested_by="agent.local",
            observation_version="v1",
        ),
    )
    assert decision.decision == PolicyDecisionKind.REQUIRE_APPROVAL
