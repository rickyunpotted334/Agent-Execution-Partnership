from __future__ import annotations

from aep.contracts.models import ActionRequest


class OperationalSystemAdapter:
    def execute(self, action: ActionRequest) -> dict[str, object]:
        return {
            "status": "success",
            "state_after": "validated_envelope",
            "response": {
                "path": "proposal->validator->envelope->supervisor->safety->actuator",
                "operation": action.operation,
            },
            "retryable": False,
        }
