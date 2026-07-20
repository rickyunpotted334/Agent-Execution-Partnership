# Threat Model

Primary threats:

- Prompt injection from web, files, or messages.
- Policy bypass attempts.
- Excessive privilege execution.
- Unauthorized file system and process actions.
- Stale-state action execution.
- Audit tampering.

Controls:

- Default deny and explicit capability checks.
- Prohibited-target registry.
- Path and domain allowlists.
- Observation freshness checks.
- Immutable audit digest chain verification.
