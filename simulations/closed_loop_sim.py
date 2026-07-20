"""
Closed-loop simulation: exercises the full AEP pipeline without a live server.

Run with:
    py simulations/closed_loop_sim.py
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from uuid import uuid4

# Ensure the project src is on the path when run directly
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aep.adapters.filesystem.local_fs_adapter import FilesystemAdapter
from aep.adapters.http.http_adapter import HTTPAdapter
from aep.adapters.messaging.mock_messaging_adapter import MessagingAdapter
from aep.adapters.browser.playwright_adapter import BrowserAdapter
from aep.adapters.process.process_adapter import ProcessAdapter
from aep.contracts.models import (
    ActionRequest,
    Observation,
    RiskClass,
    TaskContract,
)
from aep.execution.engine import ActionExecutionEngine
from aep.observations.service import ObservationService
from aep.telemetry.logging import configure_logging

configure_logging()


def run_simulation() -> None:
    obs_service = ObservationService()

    engine = ActionExecutionEngine(
        adapters={
            "filesystem": FilesystemAdapter(),
            "process": ProcessAdapter(),
            "http": HTTPAdapter(),
            "messaging": MessagingAdapter(),
            "browser": BrowserAdapter(),
        }
    )

    task = TaskContract(
        goal="Read a local file for simulation purposes",
        human_principal="operator",
        agent_identity="sim-agent",
        completion_criteria=["file_read_confirmed"],
        allowed_tools=["read_file"],
        idempotency_key=str(uuid4()),
    )

    observation = Observation(
        task_id=task.task_id,
        source_adapter="filesystem",
        source_type="file_listing",
        freshness_deadline=datetime.now(UTC) + timedelta(minutes=5),
        state_version="v_sim_1",
        confidence=0.95,
        raw_evidence_reference="local://sim/state",
        correlation_id=str(uuid4()),
    )
    obs_service.record(observation)

    action = ActionRequest(
        task_id=task.task_id,
        operation="read_file",
        channel="filesystem",
        target={"path": "examples/tmp/ok.txt"},
        risk_class=RiskClass.READ_ONLY,
        idempotency_key=str(uuid4()),
        requested_by="sim-agent",
        observation_version="v_sim_1",
    )

    obs = obs_service.get_by_state_version(task.task_id, action.observation_version)
    if obs is None or not obs_service.is_fresh(obs):
        print(json.dumps({"sim": "aborted", "reason": "stale_or_missing_observation"}))
        sys.exit(1)

    evidence, verification = engine.execute(task, action, correlation_id=str(uuid4()))

    print(
        json.dumps(
            {
                "sim": "complete",
                "verdict": verification.verdict,
                "status": evidence.status,
                "expected_effects_verified": verification.expected_effects_verified,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_simulation()
