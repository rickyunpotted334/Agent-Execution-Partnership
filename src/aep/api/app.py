from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from aep.adapters.browser.playwright_adapter import BrowserAdapter
from aep.adapters.filesystem.local_fs_adapter import FilesystemAdapter
from aep.adapters.http.http_adapter import HTTPAdapter
from aep.adapters.messaging.mock_messaging_adapter import MessagingAdapter
from aep.adapters.process.process_adapter import ProcessAdapter
from aep.approvals.service import ApprovalService
from aep.audit.ledger import AuditLedger
from aep.config.settings import get_settings
from aep.contracts.models import (
    ActionRequest,
    ExecutionEvidence,
    Observation,
    PolicyDecision,
    TaskContract,
    VerificationResult,
)
from aep.execution.engine import ActionExecutionEngine
from aep.persistence.db import init_db
from aep.policy.engine import PolicyEngine

settings = get_settings()

app = FastAPI(
    title="Agent Execution Partnership AEE API",
    description="An open-source control plane for authorized, observable, bounded, and verifiable AI agent actions.",
    version="0.1.0",
)

approval_service = ApprovalService()
policy_engine = PolicyEngine()
audit = AuditLedger()
tasks: dict[str, TaskContract] = {}
observations: dict[str, Observation] = {}

engine = ActionExecutionEngine(
    adapters={
        "filesystem": FilesystemAdapter(),
        "process": ProcessAdapter(),
        "http": HTTPAdapter(),
        "messaging": MessagingAdapter(),
        "browser": BrowserAdapter(),
    }
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aee"}


@app.get("/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "adapters": sorted(engine.adapters.keys()),
        "risk_classes": [
            "READ_ONLY",
            "REVERSIBLE_WRITE",
            "CONSEQUENTIAL_WRITE",
            "FINANCIAL",
            "PRIVILEGED_ADMINISTRATIVE",
            "PHYSICAL_OR_SAFETY_RELATED",
        ],
        "destructive_enabled": settings.enable_destructive,
    }


@app.post("/tasks", response_model=TaskContract)
def create_task(task: TaskContract) -> TaskContract:
    tasks[task.task_id] = task
    audit.append(
        actor=task.agent_identity,
        task=task.task_id,
        action="-",
        event_type="task_created",
        payload={"goal": task.goal},
        correlation_id=str(uuid4()),
    )
    return task


@app.get("/tasks/{task_id}", response_model=TaskContract)
def get_task(task_id: str) -> TaskContract:
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="task_not_found")
    return tasks[task_id]


@app.post("/observations", response_model=Observation)
def submit_observation(observation: Observation) -> Observation:
    observations[observation.observation_id] = observation
    return observation


@app.post("/policy/evaluate", response_model=PolicyDecision)
def evaluate_policy(action: ActionRequest) -> PolicyDecision:
    task = tasks.get(action.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task_not_found")
    return policy_engine.evaluate(task, action)


@app.post("/actions/propose", response_model=ActionRequest)
def propose_action(action: ActionRequest) -> ActionRequest:
    if action.task_id not in tasks:
        raise HTTPException(status_code=404, detail="task_not_found")
    return action


@app.post("/approvals")
def create_approval(action_id: str, required_by: str) -> dict[str, str]:
    approval_id = str(uuid4())
    approval_service.create(approval_id, action_id, required_by)
    return {"approval_id": approval_id}


@app.get("/approvals")
def list_approvals() -> list[dict[str, str | None]]:
    return [a.__dict__ for a in approval_service.list()]


@app.post("/approvals/{approval_id}/decision")
def decide_approval(approval_id: str, decision: str, decided_by: str) -> dict[str, str | None]:
    if decision not in {"approved", "denied"}:
        raise HTTPException(status_code=400, detail="invalid_decision")
    return approval_service.decide(approval_id, decision, decided_by).__dict__


@app.post("/actions/execute")
def execute_action(action: ActionRequest) -> dict[str, ExecutionEvidence | VerificationResult]:
    task = tasks.get(action.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task_not_found")
    observation_is_stale = False
    for obs in observations.values():
        if obs.task_id == action.task_id and obs.state_version == action.observation_version:
            observation_is_stale = obs.freshness_deadline < datetime.now(UTC)
            break
    if observation_is_stale:
        raise HTTPException(status_code=409, detail="stale_observation")

    evidence, verification = engine.execute(task, action, correlation_id=str(uuid4()))
    return {"evidence": evidence, "verification": verification}


@app.post("/verify", response_model=VerificationResult)
def verify(evidence: ExecutionEvidence) -> VerificationResult:
    return engine.verifier.verify(evidence, expected_effects=[])


@app.post("/cancel/{task_id}")
def cancel(task_id: str) -> dict[str, str]:
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="task_not_found")
    audit.append(
        actor="operator",
        task=task_id,
        action="-",
        event_type="task_cancelled",
        payload={"reason": "manual"},
        correlation_id=str(uuid4()),
    )
    return {"status": "cancelled", "task_id": task_id}


@app.get("/audit/verify")
def verify_audit() -> dict[str, bool]:
    return {"ok": audit.verify_chain()}


@app.get("/metrics")
def metrics() -> dict[str, int]:
    return {
        "tasks": len(tasks),
        "observations": len(observations),
        "approvals": len(approval_service.list()),
    }


@app.post("/experiments/register")
def register_experiment(hypothesis: str) -> dict[str, str]:
    return {
        "experiment_id": str(uuid4()),
        "hypothesis": hypothesis,
        "registered_at": datetime.now(UTC).isoformat(),
    }


@app.get("/experiments/results")
def experiment_results() -> dict[str, object]:
    return {
        "status": "stub",
        "dataset_version": "aee-dataset-v1",
        "updated_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
    }
