import json
from pathlib import Path


def test_public_schemas_exist() -> None:
    root = Path("schemas")
    expected = [
        "task_contract.schema.json",
        "observation.schema.json",
        "action_request.schema.json",
        "policy_decision.schema.json",
        "execution_evidence.schema.json",
        "verification_result.schema.json",
    ]
    for name in expected:
        p = root / name
        assert p.exists()
        _ = json.loads(p.read_text(encoding="utf-8"))
