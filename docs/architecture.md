# Architecture

Agent Execution Partnership AEE implements a closed-loop control plane:

1. Task contract creation.
2. Observation ingestion and freshness checks.
3. Single bounded action proposal.
4. Policy evaluation (default deny).
5. Approval gate for high-risk actions.
6. Adapter execution.
7. Verification and recovery planning.
8. Immutable audit recording.

Core layers:

- Contracts: strongly typed Pydantic models.
- Policy: PDP and PEP behavior.
- Execution: deterministic lifecycle and transition validation.
- Adapters: capability-bounded channels.
- Audit: append-only hash chain.
