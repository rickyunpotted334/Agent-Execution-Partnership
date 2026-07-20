from aep.adapters.browser.playwright_adapter import BrowserAdapter
from aep.contracts.models import ActionRequest, RiskClass


def test_browser_read_demo() -> None:
    adapter = BrowserAdapter()
    action = ActionRequest(
        task_id="t1",
        operation="read_page",
        channel="browser",
        target={"url": "http://127.0.0.1:8765"},
        risk_class=RiskClass.READ_ONLY,
        idempotency_key="k1",
        requested_by="agent.local",
        observation_version="v1",
    )
    result = adapter.execute(action)
    assert result["status"] == "success"
