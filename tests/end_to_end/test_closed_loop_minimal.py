from datetime import UTC, datetime, timedelta

from aep.contracts.models import ActionRequest, Observation, RiskClass, TaskContract
from aep.execution.engine import ActionExecutionEngine


class StubAdapter:
    def execute(self, action: ActionRequest) -> dict[str, object]:
        return {
            "status": "success",
            "state_after": "ok",
            "logs_reference": "stub://ok",
        }


def test_closed_loop_success() -> None:
    task = TaskContract(
        goal="read",
        human_principal="alice",
        agent_identity="agent.local",
        completion_criteria=["ok"],
        idempotency_key="k1",
        allowed_tools=["read_page"],
    )
    _observation = Observation(
        task_id=task.task_id,
        source_adapter="browser",
        source_type="dom",
        freshness_deadline=datetime.now(UTC) + timedelta(minutes=2),
        state_version="v1",
        confidence=0.9,
        raw_evidence_reference="local://evidence",
        correlation_id="c1",
    )
    action = ActionRequest(
        task_id=task.task_id,
        operation="read_page",
        channel="browser",
        target={"url": "http://127.0.0.1:8765"},
        risk_class=RiskClass.READ_ONLY,
        idempotency_key="k1",
        requested_by="agent.local",
        observation_version="v1",
    )
    engine = ActionExecutionEngine({"browser": StubAdapter()})
    evidence, verification = engine.execute(task, action, correlation_id="c1")
    assert evidence.status == "success"
    assert verification.verdict.value == "verified"
