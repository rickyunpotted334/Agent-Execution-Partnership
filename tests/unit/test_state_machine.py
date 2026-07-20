from aep.actions.state_machine import InvalidTransitionError, validate_sequence, validate_transition
from aep.contracts.models import ActionStatus


def test_valid_transition() -> None:
    validate_transition(ActionStatus.PROPOSED, ActionStatus.VALIDATING)


def test_invalid_transition_fails_closed() -> None:
    try:
        validate_transition(ActionStatus.PROPOSED, ActionStatus.EXECUTING)
    except InvalidTransitionError:
        return
    raise AssertionError("expected invalid transition")


def test_sequence_validation() -> None:
    validate_sequence(
        [
            ActionStatus.PROPOSED,
            ActionStatus.VALIDATING,
            ActionStatus.AUTHORIZED,
            ActionStatus.PRECONDITION_CHECK,
            ActionStatus.READY,
            ActionStatus.EXECUTING,
            ActionStatus.VERIFYING,
            ActionStatus.VERIFIED,
        ]
    )
