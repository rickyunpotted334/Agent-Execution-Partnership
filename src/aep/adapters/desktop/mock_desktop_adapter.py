from __future__ import annotations

from aep.contracts.models import ActionRequest

PROHIBITED_TARGETS = {
    "copilot_chat",
    "ide_chat_panel",
    "build_terminal",
    "auth_dialog",
    "password_manager",
    "security_software",
    "shutdown_controls",
}


class DesktopAdapter:
    def execute(self, action: ActionRequest) -> dict[str, object]:
        target = str(action.target.get("control", ""))
        if target in PROHIBITED_TARGETS:
            return {"status": "failed", "error": "prohibited_target", "retryable": False}
        return {
            "status": "success",
            "state_after": "desktop_action_simulated",
            "response": {"target": target},
        }
