from __future__ import annotations

from aep.contracts.models import ActionRequest


class InfrastructureAdapter:
    def execute(self, action: ActionRequest) -> dict[str, object]:
        return {
            "status": "success",
            "state_after": "dry_run_only",
            "response": {
                "operation": action.operation,
                "target": action.target,
                "dry_run": True,
            },
            "retryable": False,
            "compensation_available": True,
        }
