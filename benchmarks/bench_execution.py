"""
Execution-engine throughput benchmark.

Run with:
    py benchmarks/bench_execution.py

Reports iterations-per-second for the core execute() path using only
in-process adapters (no network, no disk I/O beyond the audit ledger).
"""
from __future__ import annotations

import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from aep.adapters.filesystem.local_fs_adapter import FilesystemAdapter
from aep.adapters.http.http_adapter import HTTPAdapter
from aep.adapters.messaging.mock_messaging_adapter import MessagingAdapter
from aep.adapters.browser.playwright_adapter import BrowserAdapter
from aep.adapters.process.process_adapter import ProcessAdapter
from aep.contracts.models import ActionRequest, RiskClass, TaskContract
from aep.execution.engine import ActionExecutionEngine


def _make_engine() -> ActionExecutionEngine:
    return ActionExecutionEngine(
        adapters={
            "filesystem": FilesystemAdapter(),
            "process": ProcessAdapter(),
            "http": HTTPAdapter(),
            "messaging": MessagingAdapter(),
            "browser": BrowserAdapter(),
        }
    )


def _make_task() -> TaskContract:
    return TaskContract(
        goal="benchmark read",
        human_principal="bench",
        agent_identity="bench-agent",
        completion_criteria=["done"],
        allowed_tools=["read_file"],
        idempotency_key=str(uuid4()),
    )


def bench_execute(iterations: int = 200) -> None:
    engine = _make_engine()
    task = _make_task()

    # Warm-up
    for _ in range(5):
        action = ActionRequest(
            task_id=task.task_id,
            operation="read_file",
            channel="filesystem",
            target={"path": "examples/tmp/ok.txt"},
            risk_class=RiskClass.READ_ONLY,
            idempotency_key=str(uuid4()),
            requested_by="bench-agent",
            observation_version="v1",
        )
        engine.execute(task, action, correlation_id=str(uuid4()))

    start = time.perf_counter()
    for _ in range(iterations):
        action = ActionRequest(
            task_id=task.task_id,
            operation="read_file",
            channel="filesystem",
            target={"path": "examples/tmp/ok.txt"},
            risk_class=RiskClass.READ_ONLY,
            idempotency_key=str(uuid4()),
            requested_by="bench-agent",
            observation_version="v1",
        )
        engine.execute(task, action, correlation_id=str(uuid4()))
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed
    print(f"bench_execute: {iterations} iterations in {elapsed:.3f}s  "
          f"→  {ops_per_sec:.1f} ops/sec")


def bench_policy_evaluate(iterations: int = 1000) -> None:
    from aep.policy.engine import PolicyEngine

    engine = PolicyEngine()
    task = _make_task()

    start = time.perf_counter()
    for _ in range(iterations):
        action = ActionRequest(
            task_id=task.task_id,
            operation="read_file",
            channel="filesystem",
            target={"path": "examples/tmp/ok.txt"},
            risk_class=RiskClass.READ_ONLY,
            idempotency_key=str(uuid4()),
            requested_by="bench-agent",
            observation_version="v1",
        )
        engine.evaluate(task, action)
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed
    print(f"bench_policy_evaluate: {iterations} iterations in {elapsed:.3f}s  "
          f"→  {ops_per_sec:.1f} ops/sec")


if __name__ == "__main__":
    bench_policy_evaluate()
    bench_execute()
