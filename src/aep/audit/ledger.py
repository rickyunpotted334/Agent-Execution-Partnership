from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from aep.config.settings import get_settings
from aep.security.redaction import redact_payload


@dataclass
class AuditEvent:
    event_id: str
    previous_event_digest: str
    event_digest: str
    timestamp: str
    actor: str
    task: str
    action: str
    event_type: str
    redacted_payload: dict[str, object]
    correlation_id: str


class AuditLedger:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.path = Path(self.settings.audit_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_digest(self) -> str:
        if not self.path.exists():
            return ""
        text = self.path.read_text(encoding="utf-8").strip()
        if not text:
            return ""
        last = text.splitlines()[-1]
        digest = json.loads(last).get("event_digest", "")
        return str(digest)

    def append(
        self,
        *,
        actor: str,
        task: str,
        action: str,
        event_type: str,
        payload: dict[str, object],
        correlation_id: str,
    ) -> AuditEvent:
        prev = self._last_digest()
        event_id = str(uuid4())
        redacted = redact_payload(payload)
        digest_input = json.dumps(
            {
                "event_id": event_id,
                "previous": prev,
                "payload": redacted,
                "event_type": event_type,
                "correlation_id": correlation_id,
            },
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(digest_input).hexdigest()
        event = AuditEvent(
            event_id=event_id,
            previous_event_digest=prev,
            event_digest=digest,
            timestamp=datetime.now(UTC).isoformat(),
            actor=actor,
            task=task,
            action=action,
            event_type=event_type,
            redacted_payload=redacted,
            correlation_id=correlation_id,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.__dict__, sort_keys=True) + "\n")
        return event

    def verify_chain(self) -> bool:
        if not self.path.exists():
            return True
        lines = self.path.read_text(encoding="utf-8").splitlines()
        prev = ""
        for line in lines:
            rec = json.loads(line)
            if rec["previous_event_digest"] != prev:
                return False
            digest_input = json.dumps(
                {
                    "event_id": rec["event_id"],
                    "previous": rec["previous_event_digest"],
                    "payload": rec["redacted_payload"],
                    "event_type": rec["event_type"],
                    "correlation_id": rec["correlation_id"],
                },
                sort_keys=True,
            ).encode("utf-8")
            expected = hashlib.sha256(digest_input).hexdigest()
            if rec["event_digest"] != expected:
                return False
            prev = rec["event_digest"]
        return True
