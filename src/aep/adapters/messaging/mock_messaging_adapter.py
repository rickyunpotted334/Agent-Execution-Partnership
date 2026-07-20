from __future__ import annotations

import hashlib

from aep.contracts.models import ActionRequest


class MessagingAdapter:
    def __init__(self) -> None:
        self.sent_ids: set[str] = set()

    def execute(self, action: ActionRequest) -> dict[str, object]:
        recipient = str(action.arguments.get("recipient", ""))
        mode = str(action.arguments.get("mode", "draft"))
        internal = recipient.endswith("@example.local")
        dedupe_key = f"{recipient}:{action.idempotency_key}"

        if dedupe_key in self.sent_ids:
            return {"status": "failed", "error": "duplicate_send_blocked", "retryable": False}

        if mode != "draft":
            return {"status": "failed", "error": "draft_only_mode", "retryable": False}

        if not internal:
            return {"status": "failed", "error": "external_recipient_blocked", "retryable": False}

        self.sent_ids.add(dedupe_key)
        content = str(action.arguments.get("content", ""))
        return {
            "status": "success",
            "state_after": "draft_created",
            "logs_reference": "msg://mock",
            "response": {
                "draft_id": hashlib.sha256(dedupe_key.encode("utf-8")).hexdigest()[:16],
                "recipient": recipient,
                "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            },
            "retryable": False,
        }
