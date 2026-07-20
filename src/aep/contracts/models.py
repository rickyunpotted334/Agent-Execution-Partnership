from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class RiskClass(str, Enum):
    READ_ONLY = "READ_ONLY"
    REVERSIBLE_WRITE = "REVERSIBLE_WRITE"
    CONSEQUENTIAL_WRITE = "CONSEQUENTIAL_WRITE"
    FINANCIAL = "FINANCIAL"
    PRIVILEGED_ADMINISTRATIVE = "PRIVILEGED_ADMINISTRATIVE"
    PHYSICAL_OR_SAFETY_RELATED = "PHYSICAL_OR_SAFETY_RELATED"


class PolicyDecisionKind(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"
    REQUIRE_MORE_OBSERVATION = "require_more_observation"
    REQUIRE_LOWER_RISK_ALTERNATIVE = "require_lower_risk_alternative"


class VerificationVerdict(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    UNCERTAIN = "uncertain"
    STALE = "stale"
    COMPENSATED = "compensated"
    ESCALATED = "escalated"


class ActionStatus(str, Enum):
    PROPOSED = "PROPOSED"
    VALIDATING = "VALIDATING"
    DENIED = "DENIED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    AUTHORIZED = "AUTHORIZED"
    PRECONDITION_CHECK = "PRECONDITION_CHECK"
    READY = "READY"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    RECOVERING = "RECOVERING"
    COMPENSATING = "COMPENSATING"
    CANCELLED = "CANCELLED"
    ESCALATED = "ESCALATED"


class TaskContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    human_principal: str
    agent_identity: str
    completion_criteria: list[str]
    constraints: list[str] = Field(default_factory=list)
    prohibited_effects: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    allowed_resources: list[str] = Field(default_factory=list)
    deadline: datetime | None = None
    risk_budget: float = Field(default=100.0, ge=0)
    approval_requirements: list[str] = Field(default_factory=list)
    data_classification: str = "internal"
    idempotency_key: str
    cancellation_conditions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    observation_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    source_adapter: str
    source_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    freshness_deadline: datetime
    state_version: str
    provenance: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    raw_evidence_reference: str
    normalized_state: dict[str, Any] = Field(default_factory=dict)
    sensitive_fields_redacted: bool = True
    correlation_id: str


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    operation: str
    channel: str
    target: dict[str, Any]
    arguments: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    expected_effects: list[str] = Field(default_factory=list)
    prohibited_effects: list[str] = Field(default_factory=list)
    risk_class: RiskClass
    timeout_ms: int = Field(default=10000, ge=1, le=120000)
    idempotency_key: str
    required_capabilities: list[str] = Field(default_factory=list)
    requested_by: str
    observation_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    action_id: str
    decision: PolicyDecisionKind
    reason_codes: list[str] = Field(default_factory=list)
    required_approvals: list[str] = Field(default_factory=list)
    constraints_added: list[str] = Field(default_factory=list)
    expiration: datetime | None = None
    policy_version: str
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExecutionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    action_id: str
    adapter: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    status: str
    request_digest: str
    response_digest: str
    artifacts: list[str] = Field(default_factory=list)
    state_before_reference: str
    state_after_reference: str
    resource_usage: dict[str, Any] = Field(default_factory=dict)
    side_effects_detected: list[str] = Field(default_factory=list)
    logs_reference: str
    error: str | None = None
    retryable: bool = False
    compensation_available: bool = False


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verification_id: str = Field(default_factory=lambda: str(uuid4()))
    action_id: str
    expected_effects_verified: bool
    prohibited_effects_absent: bool
    completion_criteria_progress: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_references: list[str] = Field(default_factory=list)
    verdict: VerificationVerdict
    recommended_next_state: str


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_baseline: str
    hypothesis: str
    changed_files: list[str]
    model_version: str
    dataset_version: str
    random_seeds: list[int]
    hardware: str
    runtime_seconds: int
    metrics: dict[str, float]
    safety_results: dict[str, bool]
    decision: str
    failure_analysis: str | None = None
    artifact_hashes: list[str] = Field(default_factory=list)
    # BPB metric from autoresearch-master (lower = better; None for non-training experiments)
    val_bpb: float | None = None
