from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class ApprovalRecord:
    approval_id: str
    action_id: str
    required_by: str
    status: str
    decided_by: str | None = None
    decided_at: datetime | None = None


class ApprovalService:
    def __init__(self) -> None:
        self._approvals: dict[str, ApprovalRecord] = {}

    def create(self, approval_id: str, action_id: str, required_by: str) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=approval_id,
            action_id=action_id,
            required_by=required_by,
            status="pending",
        )
        self._approvals[approval_id] = record
        return record

    def list(self) -> list[ApprovalRecord]:
        return list(self._approvals.values())

    def decide(self, approval_id: str, decision: str, decided_by: str) -> ApprovalRecord:
        record = self._approvals[approval_id]
        record.status = decision
        record.decided_by = decided_by
        record.decided_at = datetime.now(UTC)
        return record
