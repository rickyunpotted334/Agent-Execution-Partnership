from aep.contracts.models import ActionRequest, RiskClass, TaskContract
from aep.execution.engine import ActionExecutionEngine


class FailingAdapter:
    def execute(self, action: ActionRequest) -> dict[str, object]:
        return {
            "status": "failed",
            "error": "simulated_failure",
            "retryable": True,
            "compensation_available": True,
            "logs_reference": "stub://fail",
        }


def test_financial_action_awaits_approval() -> None:
    engine = ActionExecutionEngine({"browser": FailingAdapter()})
    task = TaskContract(
        goal="pay",
        human_principal="alice",
        agent_identity="agent.local",
        completion_criteria=["paid"],
        idempotency_key="k1",
        allowed_tools=["pay"],
    )
    action = ActionRequest(
        task_id=task.task_id,
        operation="pay",
        channel="browser",
        target={"id": "x"},
        risk_class=RiskClass.FINANCIAL,
        idempotency_key="k2",
        requested_by="agent.local",
        observation_version="v1",
    )
    evidence, _ = engine.execute(task, action, correlation_id="c1")
    assert evidence.status == "awaiting_approval"


def test_failed_action_enters_recovery_path() -> None:
    engine = ActionExecutionEngine({"browser": FailingAdapter()})
    task = TaskContract(
        goal="change",
        human_principal="alice",
        agent_identity="agent.local",
        completion_criteria=["changed"],
        idempotency_key="k3",
        allowed_tools=["change"],
    )
    action = ActionRequest(
        task_id=task.task_id,
        operation="change",
        channel="browser",
        target={"id": "y"},
        risk_class=RiskClass.REVERSIBLE_WRITE,
        idempotency_key="k4",
        requested_by="agent.local",
        observation_version="v1",
    )
    evidence, verification = engine.execute(task, action, correlation_id="c2")
    assert evidence.status == "failed"
    assert verification.verdict.value == "failed"
