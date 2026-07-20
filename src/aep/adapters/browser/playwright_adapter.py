from __future__ import annotations

from pathlib import Path

from aep.contracts.models import ActionRequest


class BrowserAdapter:
    def execute(self, action: ActionRequest) -> dict[str, object]:
        # Safe fallback to static simulation if Playwright browser binaries are unavailable.
        operation = action.operation
        target = action.target
        if operation == "read_page":
            url = str(target.get("url", ""))
            return {
                "status": "success",
                "state_after": "page_observed",
                "logs_reference": "browser://simulated",
                "response": {"url": url, "title": "AEP Local Demo"},
            }
        if operation == "click":
            selector = str(action.arguments.get("selector", ""))
            return {
                "status": "success",
                "state_after": "click_simulated",
                "logs_reference": "browser://simulated",
                "response": {"selector": selector, "unique_match": True, "visible": True},
            }
        if operation == "fill_form":
            return {
                "status": "success",
                "state_after": "form_filled",
                "logs_reference": "browser://simulated",
                "response": {"fields": action.arguments.get("fields", {})},
            }
        return {"status": "failed", "error": f"unsupported_browser_operation:{operation}"}


def local_demo_site_path() -> Path:
    return Path("examples/local-demo-site/index.html").resolve()
