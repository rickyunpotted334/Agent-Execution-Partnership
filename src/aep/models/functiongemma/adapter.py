from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from aep.contracts.models import ActionRequest, TaskContract


@dataclass
class FunctionGemmaResult:
    kind: str
    payload: dict[str, Any]
    confidence: float


class FunctionGemmaAdapter:
    def __init__(self, model_version: str = "functiongemma-baseline-v1") -> None:
        self.model_version = model_version

    def decide(self, task: TaskContract, tool_defs: list[dict[str, Any]]) -> FunctionGemmaResult:
        if not tool_defs:
            return FunctionGemmaResult("request_more_observation", {}, 0.5)
        return FunctionGemmaResult(
            kind="action_request",
            payload={
                "task_id": task.task_id,
                "operation": "noop",
                "channel": "simulated",
                "target": {"kind": "none"},
                "arguments": {},
                "preconditions": [],
                "expected_effects": ["no_change"],
                "prohibited_effects": [],
                "risk_class": "READ_ONLY",
                "timeout_ms": 1000,
                "idempotency_key": task.idempotency_key,
                "required_capabilities": [],
                "requested_by": task.agent_identity,
                "observation_version": "v1",
                "metadata": {"model": self.model_version},
            },
            confidence=0.7,
        )

    def parse_action(self, payload: dict[str, Any]) -> ActionRequest:
        try:
            return ActionRequest.model_validate(payload)
        except ValidationError as exc:
            raise ValueError("Malformed FunctionGemma function call output") from exc
