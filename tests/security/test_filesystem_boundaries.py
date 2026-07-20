from pathlib import Path

import pytest

from aep.adapters.filesystem.local_fs_adapter import FilesystemAdapter
from aep.contracts.models import ActionRequest, RiskClass


def test_blocks_outside_allowed_root() -> None:
    adapter = FilesystemAdapter()
    action = ActionRequest(
        task_id="t1",
        operation="read_file",
        channel="filesystem",
        target={"path": "C:/Windows/System32/drivers/etc/hosts"},
        risk_class=RiskClass.READ_ONLY,
        idempotency_key="k1",
        requested_by="agent.local",
        observation_version="v1",
    )
    with pytest.raises(PermissionError):
        adapter.execute(action)


def test_write_inside_allowed_root() -> None:
    adapter = FilesystemAdapter()
    root = Path("E:/Agent-Execution-Partnership/agent-execution-partnership/examples/tmp")
    root.mkdir(parents=True, exist_ok=True)
    p = root / "ok.txt"
    action = ActionRequest(
        task_id="t2",
        operation="create_file",
        channel="filesystem",
        target={"path": str(p)},
        arguments={"content": "ok"},
        risk_class=RiskClass.REVERSIBLE_WRITE,
        idempotency_key="k2",
        requested_by="agent.local",
        observation_version="v1",
    )
    res = adapter.execute(action)
    assert res["status"] == "success"
    assert p.exists()
