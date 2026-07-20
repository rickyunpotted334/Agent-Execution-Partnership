from __future__ import annotations

from urllib.parse import urlparse

import httpx

from aep.config.settings import get_settings
from aep.contracts.models import ActionRequest


class HTTPAdapter:
    SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

    def __init__(self) -> None:
        self.settings = get_settings()

    def execute(self, action: ActionRequest) -> dict[str, object]:
        method = str(action.arguments.get("method", "GET")).upper()
        url = str(action.target.get("url", ""))
        parsed = urlparse(url)
        if (
            parsed.hostname is None
            or parsed.hostname.lower() not in self.settings.http_allow_domains
        ):
            return {"status": "failed", "error": "domain_not_allowed", "retryable": False}

        if method not in self.SAFE_METHODS and not self.settings.enable_destructive:
            return {"status": "failed", "error": "unsafe_method_disabled", "retryable": False}

        headers = {"x-idempotency-key": action.idempotency_key}
        with httpx.Client(timeout=min(action.timeout_ms / 1000.0, 10.0)) as client:
            resp = client.request(method, url, headers=headers)
        return {
            "status": "success" if resp.status_code < 400 else "failed",
            "state_after": "response_received",
            "logs_reference": "http://local",
            "response": {"status_code": resp.status_code, "content_length": len(resp.content)},
            "retryable": method in self.SAFE_METHODS and resp.status_code >= 500,
        }
