from aep.contracts.models import ExperimentRecord
from aep.research.autoresearch.harness import AutoresearchHarness


def test_failed_safety_reverts() -> None:
    h = AutoresearchHarness()
    r = ExperimentRecord(
        parent_baseline="b1",
        hypothesis="h",
        changed_files=["x"],
        model_version="m1",
        dataset_version="d1",
        random_seeds=[1],
        hardware="cpu",
        runtime_seconds=1,
        metrics={"end_to_end_completion_rate": 0.9},
        safety_results={"policy": False},
        decision="pending",
    )
    d = h.evaluate(r)
    assert not d.retain


def test_success_retain() -> None:
    h = AutoresearchHarness()
    r = ExperimentRecord(
        parent_baseline="b1",
        hypothesis="h",
        changed_files=["x"],
        model_version="m1",
        dataset_version="d1",
        random_seeds=[1],
        hardware="cpu",
        runtime_seconds=1,
        metrics={"end_to_end_completion_rate": 0.9},
        safety_results={"policy": True},
        decision="pending",
    )
    d = h.evaluate(r)
    assert d.retain
