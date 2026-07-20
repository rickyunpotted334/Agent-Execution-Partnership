from pathlib import Path

from aep.audit.ledger import AuditLedger


def test_audit_chain_verification() -> None:
    ledger = AuditLedger()
    Path(ledger.path).write_text("", encoding="utf-8")
    ledger.append(
        actor="agent.local",
        task="t1",
        action="a1",
        event_type="test",
        payload={"x": 1, "token": "dont-log"},
        correlation_id="c1",
    )
    ledger.append(
        actor="agent.local",
        task="t1",
        action="a2",
        event_type="test",
        payload={"x": 2},
        correlation_id="c2",
    )
    assert ledger.verify_chain()
