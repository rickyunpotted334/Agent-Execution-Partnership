from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from aep.api.app import app
from aep.contracts.models import ActionRequest, Observation, RiskClass, TaskContract


def test_stale_observation_is_rejected() -> None:
    client = TestClient(app)
    task = TaskContract(
        goal="read",
        human_principal="alice",
        agent_identity="agent.local",
        completion_criteria=["ok"],
        idempotency_key="k1",
        allowed_tools=["read_page"],
    )
    _ = client.post("/tasks", json=task.model_dump(mode="json"))

    observation = Observation(
        task_id=task.task_id,
        source_adapter="browser",
        source_type="dom",
        freshness_deadline=datetime.now(UTC) - timedelta(seconds=1),
        state_version="v_stale",
        confidence=0.9,
        raw_evidence_reference="local://obs",
        correlation_id="c1",
    )
    _ = client.post("/observations", json=observation.model_dump(mode="json"))

    action = ActionRequest(
        task_id=task.task_id,
        operation="read_page",
        channel="browser",
        target={"url": "http://127.0.0.1:8765"},
        risk_class=RiskClass.READ_ONLY,
        idempotency_key="k2",
        requested_by="agent.local",
        observation_version="v_stale",
    )
    resp = client.post("/actions/execute", json=action.model_dump(mode="json"))
    assert resp.status_code == 409
