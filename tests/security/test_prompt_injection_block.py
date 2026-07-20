from aep.contracts.models import ActionRequest, RiskClass
from aep.models.functiongemma.adapter import FunctionGemmaAdapter


def test_malformed_output_rejected() -> None:
    adapter = FunctionGemmaAdapter()
    bad = {
        "task_id": "x",
        "operation": "read",
        "channel": "browser",
    }
    try:
        adapter.parse_action(bad)
    except ValueError:
        return
    raise AssertionError("malformed output should fail")


def test_typed_action_accepts_valid_schema() -> None:
    adapter = FunctionGemmaAdapter()
    ok = ActionRequest(
        task_id="t1",
        operation="read_page",
        channel="browser",
        target={"url": "http://127.0.0.1:8765"},
        risk_class=RiskClass.READ_ONLY,
        idempotency_key="k1",
        requested_by="agent.local",
        observation_version="v1",
    )
    parsed = adapter.parse_action(ok.model_dump())
    assert parsed.operation == "read_page"
