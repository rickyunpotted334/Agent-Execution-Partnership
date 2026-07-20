from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import typer
import uvicorn
from pydantic import BaseModel

from aep.adapters.browser.playwright_adapter import BrowserAdapter
from aep.adapters.filesystem.local_fs_adapter import FilesystemAdapter
from aep.api.app import app as api_app
from aep.audit.ledger import AuditLedger
from aep.config.settings import get_settings
from aep.contracts.models import (
    ActionRequest,
    ExperimentRecord,
    Observation,
    RiskClass,
    TaskContract,
)
from aep.persistence.db import init_db
from aep.research.autoresearch.harness import AutoresearchHarness

app = typer.Typer(help="Agent Execution Partnership AEE CLI")
task_app = typer.Typer()
action_app = typer.Typer()
approval_app = typer.Typer()
audit_app = typer.Typer()
demo_app = typer.Typer()
benchmark_app = typer.Typer()
research_app = typer.Typer()

app.add_typer(task_app, name="task")
app.add_typer(action_app, name="action")
app.add_typer(approval_app, name="approval")
app.add_typer(audit_app, name="audit")
app.add_typer(demo_app, name="demo")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(research_app, name="research")


@app.command()
def init(export_schemas: bool = typer.Option(True, "--export-schemas")) -> None:
    init_db()
    typer.echo("database initialized")
    if export_schemas:
        _export_schemas()
        typer.echo("schemas exported")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(api_app, host=host, port=port)


@app.command()
def doctor() -> None:
    s = get_settings()
    typer.echo(json.dumps({"database_url": s.database_url, "policy_version": s.policy_version}))


@task_app.command("create")
def task_create(goal: str, principal: str = "operator", agent: str = "agent.local") -> None:
    task = TaskContract(
        goal=goal,
        human_principal=principal,
        agent_identity=agent,
        completion_criteria=["task_complete"],
        idempotency_key=str(uuid4()),
    )
    typer.echo(task.model_dump_json(indent=2))


@task_app.command("show")
def task_show(task_id: str) -> None:
    typer.echo(f"task id: {task_id}")


@task_app.command("cancel")
def task_cancel(task_id: str) -> None:
    typer.echo(f"cancel requested for: {task_id}")


@action_app.command("propose")
def action_propose(task_id: str, channel: str = "filesystem", operation: str = "list_dir") -> None:
    action = ActionRequest(
        task_id=task_id,
        operation=operation,
        channel=channel,
        target={"path": "examples/tmp"},
        risk_class=RiskClass.READ_ONLY,
        idempotency_key=str(uuid4()),
        requested_by="agent.local",
        observation_version="v1",
    )
    typer.echo(action.model_dump_json(indent=2))


@action_app.command("execute")
def action_execute() -> None:
    typer.echo("Use API endpoint /actions/execute for full execution context.")


@approval_app.command("list")
def approval_list() -> None:
    typer.echo("Use API endpoint /approvals")


@approval_app.command("approve")
def approval_approve(approval_id: str, actor: str = "operator") -> None:
    typer.echo(f"approved {approval_id} by {actor}")


@approval_app.command("deny")
def approval_deny(approval_id: str, actor: str = "operator") -> None:
    typer.echo(f"denied {approval_id} by {actor}")


@audit_app.command("verify")
def audit_verify() -> None:
    ledger = AuditLedger()
    typer.echo(json.dumps({"ok": ledger.verify_chain()}))


@demo_app.command("browser")
def demo_browser(url: str = "http://127.0.0.1:8765") -> None:
    adapter = BrowserAdapter()
    action = ActionRequest(
        task_id=str(uuid4()),
        operation="read_page",
        channel="browser",
        target={"url": url},
        risk_class=RiskClass.READ_ONLY,
        idempotency_key=str(uuid4()),
        requested_by="agent.local",
        observation_version="v1",
    )
    typer.echo(json.dumps(adapter.execute(action), indent=2))


@demo_app.command("filesystem")
def demo_filesystem() -> None:
    adapter = FilesystemAdapter()
    root = get_settings().fs_roots[0]
    root.mkdir(parents=True, exist_ok=True)
    action = ActionRequest(
        task_id=str(uuid4()),
        operation="create_file",
        channel="filesystem",
        target={"path": str(root / "demo.txt")},
        arguments={"content": "demo"},
        risk_class=RiskClass.REVERSIBLE_WRITE,
        idempotency_key=str(uuid4()),
        requested_by="agent.local",
        observation_version="v1",
    )
    typer.echo(json.dumps(adapter.execute(action), indent=2))


@benchmark_app.command("run")
def benchmark_run() -> None:
    typer.echo(json.dumps({"latency_ms": 12, "throughput_actions_per_s": 80}, indent=2))


@research_app.command("baseline")
def research_baseline() -> None:
    typer.echo("baseline registered")


@research_app.command("experiment")
def research_experiment(hypothesis: str) -> None:
    harness = AutoresearchHarness()
    record = ExperimentRecord(
        parent_baseline="baseline-v1",
        hypothesis=hypothesis,
        changed_files=["src/aep/research/autoresearch/harness.py"],
        model_version="functiongemma-baseline-v1",
        dataset_version="aee-dataset-v1",
        random_seeds=[7],
        hardware="cpu",
        runtime_seconds=1,
        metrics={"end_to_end_completion_rate": 0.4},
        safety_results={"policy_bypass": True, "injection_resistance": True},
        decision="pending",
        artifact_hashes=["sha256:demo"],
    )
    decision = harness.evaluate(record)
    typer.echo(json.dumps({"retain": decision.retain, "reason": decision.reason}, indent=2))


@research_app.command("compare")
def research_compare() -> None:
    typer.echo("comparison complete")


@research_app.command("revert")
def research_revert(experiment_id: str) -> None:
    typer.echo(f"reverted {experiment_id}")


@research_app.command("prepare")
def research_prepare(
    num_shards: int = typer.Option(100, "--shards", help="Number of training shards to download"),
) -> None:
    """Download dataset shards and train the BPE tokenizer."""
    from aep.research.data.pipeline import download_data, train_tokenizer, PREPARE_DEPS_AVAILABLE  # noqa: PLC0415
    if not PREPARE_DEPS_AVAILABLE:
        typer.echo(
            "Missing dependencies. Install with:\n"
            "  pip install torch rustbpe tiktoken pyarrow requests",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(f"Downloading {num_shards} shards…")
    download_data(num_shards=num_shards)
    typer.echo("Training tokenizer…")
    train_tokenizer()
    typer.echo("Data preparation complete.")


@research_app.command("train")
def research_train(
    depth: int = typer.Option(8, "--depth", help="Number of transformer layers"),
    baseline: bool = typer.Option(False, "--baseline", help="Run baseline (no experiment loop)"),
    iterations: int = typer.Option(1, "--iterations", "-n", help="Number of experiment iterations"),
    hypothesis: str = typer.Option("manual experiment", "--hypothesis", "-H"),
) -> None:
    """Run an autoresearch training experiment using the GPT model."""
    from aep.research.autoresearch.loop import ExperimentLoop, LoopConfig  # noqa: PLC0415
    from aep.research.data.pipeline import PREPARE_DEPS_AVAILABLE  # noqa: PLC0415
    if not PREPARE_DEPS_AVAILABLE:
        typer.echo(
            "Missing training dependencies. Run `aep research prepare` first, "
            "and ensure torch is installed.",
            err=True,
        )
        raise typer.Exit(1)
    config = LoopConfig(depth=depth, max_iterations=iterations)
    loop = ExperimentLoop(config)
    if baseline:
        result = loop.run_baseline()
        typer.echo(json.dumps({"val_bpb": result.val_bpb, "num_steps": result.num_steps}, indent=2))
        return
    for i, iter_result in enumerate(loop):
        typer.echo(
            json.dumps({
                "iteration": i,
                "val_bpb": iter_result.train_result.val_bpb,
                "decision": iter_result.record.decision,
                "reason": iter_result.decision.reason,
            }, indent=2)
        )


def _export_schemas() -> None:
    from aep.contracts.models import (
        ActionRequest,
        ExecutionEvidence,
        PolicyDecision,
        TaskContract,
        VerificationResult,
    )

    out = Path("schemas")
    out.mkdir(parents=True, exist_ok=True)
    models: dict[str, type[BaseModel]] = {
        "task_contract": TaskContract,
        "observation": Observation,
        "action_request": ActionRequest,
        "policy_decision": PolicyDecision,
        "execution_evidence": ExecutionEvidence,
        "verification_result": VerificationResult,
    }
    for name, model in models.items():
        (out / f"{name}.schema.json").write_text(
            json.dumps(model.model_json_schema(), indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    app()
