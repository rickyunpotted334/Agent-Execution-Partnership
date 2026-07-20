from __future__ import annotations

from collections.abc import Iterable

from aep.contracts.models import ActionStatus

ALLOWED_TRANSITIONS: dict[ActionStatus, set[ActionStatus]] = {
    ActionStatus.PROPOSED: {ActionStatus.VALIDATING, ActionStatus.DENIED},
    ActionStatus.VALIDATING: {
        ActionStatus.DENIED,
        ActionStatus.AWAITING_APPROVAL,
        ActionStatus.AUTHORIZED,
    },
    ActionStatus.DENIED: set(),
    ActionStatus.AWAITING_APPROVAL: {ActionStatus.AUTHORIZED, ActionStatus.CANCELLED},
    ActionStatus.AUTHORIZED: {ActionStatus.PRECONDITION_CHECK, ActionStatus.CANCELLED},
    ActionStatus.PRECONDITION_CHECK: {
        ActionStatus.READY,
        ActionStatus.FAILED,
        ActionStatus.RECOVERING,
    },
    ActionStatus.READY: {ActionStatus.EXECUTING, ActionStatus.CANCELLED},
    ActionStatus.EXECUTING: {ActionStatus.VERIFYING, ActionStatus.FAILED, ActionStatus.CANCELLED},
    ActionStatus.VERIFYING: {
        ActionStatus.VERIFIED,
        ActionStatus.FAILED,
        ActionStatus.RECOVERING,
        ActionStatus.COMPENSATING,
        ActionStatus.ESCALATED,
    },
    ActionStatus.VERIFIED: set(),
    ActionStatus.FAILED: {
        ActionStatus.RECOVERING,
        ActionStatus.COMPENSATING,
        ActionStatus.ESCALATED,
    },
    ActionStatus.RECOVERING: {
        ActionStatus.READY,
        ActionStatus.COMPENSATING,
        ActionStatus.ESCALATED,
    },
    ActionStatus.COMPENSATING: {
        ActionStatus.ESCALATED,
        ActionStatus.CANCELLED,
        ActionStatus.VERIFIED,
    },
    ActionStatus.CANCELLED: set(),
    ActionStatus.ESCALATED: set(),
}


class InvalidTransitionError(ValueError):
    pass


def validate_transition(current: ActionStatus, nxt: ActionStatus) -> None:
    if nxt not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(f"invalid transition: {current} -> {nxt}")


def validate_sequence(states: Iterable[ActionStatus]) -> None:
    seq = list(states)
    for i in range(len(seq) - 1):
        validate_transition(seq[i], seq[i + 1])
