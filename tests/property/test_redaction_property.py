from hypothesis import given
from hypothesis import strategies as st

from aep.security.redaction import redact_payload


@given(
    st.dictionaries(st.text(min_size=1, max_size=8), st.text(max_size=20), min_size=1, max_size=10)
)
def test_redaction_preserves_keys(payload: dict[str, str]) -> None:
    redacted = redact_payload(payload)
    assert set(redacted.keys()) == set(payload.keys())
