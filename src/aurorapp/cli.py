import json
import os
import uuid
from pathlib import Path
from typing import Annotated

import typer

from aurorapp.audit import AuditExporter
from aurorapp.canonical import canonical_sha256
from aurorapp.compatibility import COMPATIBILITY_LADDER
from aurorapp.controller import ControllerState, reduce_event
from aurorapp.data import (
    build_partition_plan,
    download_kernelbook,
    load_kernelbook,
    write_partition_plan,
)
from aurorapp.database import PostgresControlStore
from aurorapp.models import EvaluatorBundle, ExperimentDraft, HumanDecision, SystemMode
from aurorapp.schema import write_schemas
from aurorapp.signatures import ApprovalSigner

app = typer.Typer(no_args_is_help=True, help="Aurora++ control and audit CLI.")
config_app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
probe_app = typer.Typer(no_args_is_help=True)
system_app = typer.Typer(no_args_is_help=True)
candidate_app = typer.Typer(no_args_is_help=True)
review_app = typer.Typer(no_args_is_help=True)
bundle_app = typer.Typer(no_args_is_help=True)
audit_app = typer.Typer(no_args_is_help=True)

app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(probe_app, name="probe")
app.add_typer(system_app, name="system")
app.add_typer(candidate_app, name="candidate")
app.add_typer(review_app, name="review")
app.add_typer(bundle_app, name="bundle")
app.add_typer(audit_app, name="audit")

DEFAULT_CONFIG = Path("configs/laguna_dflash_int4.json")


def _json(value: object) -> None:
    typer.echo(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def _store(database_url: str, schema: str) -> PostgresControlStore:
    store = PostgresControlStore(database_url, schema=schema)
    store.bootstrap()
    return store


@config_app.command("check")
def config_check(
    path: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_CONFIG,
) -> None:
    draft = ExperimentDraft.model_validate_json(path.read_text(encoding="utf-8"))
    _json(
        {
            "activatable": False,
            "kind": "experiment-draft",
            "name": draft.name,
            "reason": "compatibility, data membership, runtime identity, and signature are pending",
        }
    )


@config_app.command("schemas")
def config_schemas(
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("schemas"),
) -> None:
    paths = write_schemas(output)
    _json({"count": len(paths), "paths": [str(path) for path in paths]})


@data_app.command("build")
def data_build(
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "artifacts/data/kernelbook-plan.json"
    ),
    cache_directory: Annotated[Path, typer.Option(file_okay=False)] = Path("artifacts/hf-cache"),
    seed: Annotated[int, typer.Option()] = 20260812,
) -> None:
    source = download_kernelbook(cache_directory)
    rows = load_kernelbook(source)
    plan = build_partition_plan(rows, seed)
    write_partition_plan(plan, output)
    _json(
        {
            "approved": sum(len(values) for values in plan.membership.values()),
            "group_assignment_hash": plan.group_assignment_hash,
            "output": str(output),
            "quarantined": len(plan.quarantine),
            "semantic_execution_status": plan.semantic_execution_status,
            "status": "static-plan-built",
        }
    )


@probe_app.command("compatibility")
def probe_compatibility(
    evidence_directory: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "artifacts/compatibility"
    ),
) -> None:
    evidence: dict[str, object] = {}
    for path in sorted(evidence_directory.glob("*.json")):
        try:
            evidence[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            evidence[path.name] = {"status": "invalid-json", "error": str(error)}
    target = evidence.get("target-only.json")
    target_passed = isinstance(target, dict) and target.get("status") == "passed"
    dflash = evidence.get("official-dflash.json")
    dflash_passed = isinstance(dflash, dict) and dflash.get("status") == "passed"
    completed = {
        "target-load": target_passed,
        "target-only-serving": target_passed,
        "official-dflash-load": dflash_passed,
    }
    all_steps_passed = all(completed.get(step, False) for step in COMPATIBILITY_LADDER)
    status = "passed" if all_steps_passed else "blocked"
    _json(
        {
            "completed": completed,
            "evidence_files": sorted(evidence),
            "ladder": list(COMPATIBILITY_LADDER),
            "status": status,
        }
    )
    if status != "passed":
        raise typer.Exit(2)


@system_app.command("enable")
def system_enable(
    mode: Annotated[SystemMode, typer.Option()],
    experiment_id: Annotated[str, typer.Option()],
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    if mode not in {SystemMode.SHADOW, SystemMode.AUTO_DRAFTER}:
        raise typer.BadParameter("mode must be shadow or auto-drafter")
    store = _store(database_url, schema)
    activation = store.active_signed_object("activated-experiment")
    if activation is None or activation["canonical_payload"].get("experiment_id") != experiment_id:
        _json(
            {
                "status": "blocked",
                "reason": "a matching signed activated experiment is required",
            }
        )
        raise typer.Exit(2)
    command_id = store.enqueue_command(
        experiment_id,
        "enable-shadow",
        {"mode": mode.value},
        f"cli:{uuid.uuid4()}",
    )
    _json({"command_id": command_id, "status": "queued"})


@system_app.command("drain")
def system_drain(
    experiment_id: Annotated[str, typer.Option()],
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    command_id = _store(database_url, schema).enqueue_command(
        experiment_id, "drain", {}, f"cli:{uuid.uuid4()}"
    )
    _json({"command_id": command_id, "status": "queued"})


@system_app.command("off")
def system_off(
    experiment_id: Annotated[str, typer.Option()],
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    command_id = _store(database_url, schema).enqueue_command(
        experiment_id, "off", {}, f"cli:{uuid.uuid4()}"
    )
    _json({"command_id": command_id, "status": "queued"})


@system_app.command("status")
def system_status(
    experiment_id: Annotated[str, typer.Option()],
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    store = _store(database_url, schema)
    events = store.load_events(experiment_id)
    _json(
        {
            "active_pointer": store.active_pointer(experiment_id),
            "event_count": len(events),
            "last_event": events[-1].model_dump(mode="json") if events else None,
        }
    )


@candidate_app.command("inspect")
def candidate_inspect(
    experiment_id: Annotated[str, typer.Option()],
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    rows = _store(database_url, schema).list_candidates(experiment_id)
    _json({"candidates": rows, "experiment_id": experiment_id})


@review_app.command("next")
def review_next(
    experiment_id: Annotated[str, typer.Option()],
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    _json({"review": _store(database_url, schema).next_review(experiment_id)})


@review_app.command("answer")
def review_answer(
    review_id: Annotated[int, typer.Option()],
    decision: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    value = HumanDecision.model_validate_json(decision.read_text(encoding="utf-8"))
    _store(database_url, schema).answer_review(review_id, value.model_dump(mode="json"))
    _json({"review_id": review_id, "status": "answered"})


@bundle_app.command("propose")
def bundle_propose(
    bundle: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
) -> None:
    value = EvaluatorBundle.model_validate_json(bundle.read_text(encoding="utf-8"))
    payload = value.model_dump(mode="json", exclude={"approval"})
    _json({"bundle_hash": canonical_sha256(payload), "payload": payload, "status": "valid"})


@bundle_app.command("sign")
def bundle_sign(
    bundle: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    private_key: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    signer: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option(dir_okay=False)],
) -> None:
    value = EvaluatorBundle.model_validate_json(bundle.read_text(encoding="utf-8"))
    payload = value.model_dump(mode="json", exclude={"approval"})
    approval = ApprovalSigner(private_key, signer).sign(payload)
    signed = value.model_copy(update={"approval": approval})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(signed.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _json({"bundle_hash": approval.payload_hash, "output": str(output), "status": "signed"})


@bundle_app.command("activate")
def bundle_activate(
    bundle: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    private_key: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    value = EvaluatorBundle.model_validate_json(bundle.read_text(encoding="utf-8"))
    if value.approval is None:
        raise typer.BadParameter("bundle has no approval")
    payload = value.model_dump(mode="json", exclude={"approval"})
    ApprovalSigner(private_key, value.approval.signer).verify(payload, value.approval)
    store = _store(database_url, schema)
    store.record_signed_object(
        value.approval.payload_hash,
        "evaluator-bundle",
        payload,
        value.approval.model_dump(mode="json"),
    )
    store.activate_signed_object(value.approval.payload_hash, "evaluator-bundle")
    _json({"bundle_hash": value.approval.payload_hash, "status": "active"})


@app.command("replay")
def replay(
    experiment_id: Annotated[str, typer.Option()],
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    events = _store(database_url, schema).load_events(experiment_id)
    state = ControllerState()
    for event in events:
        state = reduce_event(state, event)
    _json({"events": len(events), "state": state.__dict__})


@audit_app.command("export")
def audit_export(
    experiment_id: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option(file_okay=False)] = Path("audit-exports"),
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    store = _store(database_url, schema)
    path = AuditExporter(store).export(experiment_id, output)
    _json({"path": str(path), "status": "exported"})
