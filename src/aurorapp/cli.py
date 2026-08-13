import json
import os
import uuid
from pathlib import Path
from typing import Annotated

import typer

from aurorapp.artifacts import ContentAddressedArtifactStore
from aurorapp.audit import AuditExporter
from aurorapp.canonical import canonical_sha256
from aurorapp.compatibility import (
    COMPATIBILITY_LADDER,
    CompatibilityReport,
    CompatibilityStatus,
    CompatibilityStepResult,
)
from aurorapp.controller import ControllerState, reduce_event
from aurorapp.data import (
    build_partition_plan,
    download_kernelbook,
    load_kernelbook,
    write_partition_plan,
)
from aurorapp.database import PostgresControlStore
from aurorapp.models import (
    ArtifactRef,
    EvaluatorBundle,
    EvidenceLevel,
    ExperimentDraft,
    HumanDecision,
    SystemMode,
)
from aurorapp.schema import write_schemas
from aurorapp.sglang_contract import sampled_distribution_suite_rule
from aurorapp.signatures import ApprovalSigner
from aurorapp.source_probe import run_capture_patch_probe
from aurorapp.training_probe import run_training_model_port_probe

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


@config_app.command("sampled-distribution-rule")
def config_sampled_distribution_rule(
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
) -> None:
    rule = sampled_distribution_suite_rule()
    payload = rule.model_dump(mode="json")
    rule_hash = canonical_sha256(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rule.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _json({"rule_hash": rule_hash, "payload": payload, "output": output})


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


@probe_app.command("source-compatibility")
def probe_source_compatibility(
    config: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_CONFIG,
    evidence_directory: Annotated[Path, typer.Option(file_okay=False)] = Path(
        "artifacts/compatibility"
    ),
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    draft = ExperimentDraft.model_validate_json(config.read_text(encoding="utf-8"))
    draft_hash = canonical_sha256(draft.model_dump(mode="json"))
    result = run_capture_patch_probe(Path.cwd())
    artifact_store = ContentAddressedArtifactStore(evidence_directory / "store")
    staged = artifact_store.stage_bytes(
        "source-capture-compatibility.json",
        result.model_dump_json(indent=2).encode() + b"\n",
        producer="compatibility-source-probe",
    )
    source_artifact = artifact_store.commit(
        staged,
        loader=lambda path: bool(json.loads(path.read_text(encoding="utf-8"))),
    )
    training_model_port = run_training_model_port_probe(Path.cwd())
    training_model_staged = artifact_store.stage_bytes(
        "training-model-compatibility.json",
        training_model_port.model_dump_json(indent=2).encode() + b"\n",
        producer="compatibility-training-model-probe",
    )
    training_model_artifact = artifact_store.commit(
        training_model_staged,
        loader=lambda path: bool(json.loads(path.read_text(encoding="utf-8"))),
    )
    target = _probe_artifact(evidence_directory / "target-only-seeded.json")
    dflash = _probe_artifact(evidence_directory / "official-dflash-seeded.json")
    capture = _passing_probe_artifact(
        evidence_directory / "laguna-dflash-capture.json",
        "capture",
    )
    ingest = _passing_probe_artifact(
        evidence_directory / "specforge-batch-ingest.json",
        "ingest",
    )
    captured_optimizer = _passing_probe_artifact(
        evidence_directory / "specforge-laguna-captured-optimizer-config-preserved.json",
        "captured-optimizer",
    )
    candidate_serving = _passing_probe_artifact(
        evidence_directory / "candidate-dflash-serving-config-preserved.json",
        "candidate-serving",
    )
    parent_restore = _passing_probe_artifact(
        evidence_directory / "parent-dflash-restore.json",
        "parent-restore",
    )
    tokenizer_identity = _passing_probe_artifact(
        evidence_directory / "tokenizer-template-identity.json",
        "tokenizer-identity",
    )
    sampled_distribution = _passing_probe_artifact(
        evidence_directory / "sampled-distribution-equivalence.json",
        "sampled-distribution",
    )
    steps = _build_source_compatibility_steps(
        source_artifact,
        target,
        dflash,
        capture,
        ingest,
        training_model=training_model_artifact,
        captured_optimizer=captured_optimizer,
        candidate_serving=candidate_serving,
        parent_restore=parent_restore,
        tokenizer_identity=tokenizer_identity,
        sampled_distribution=sampled_distribution,
        training_model_compatible=training_model_port.upstream.compatible,
        training_model_port_ready=training_model_port.ready_for_physical_probe,
    )
    report = CompatibilityReport(
        identity_kind="draft",
        experiment_identity=draft_hash,
        steps=steps,
        cleanup_verified=False,
    )
    report_payload = report.model_dump(mode="json")
    report_hash = canonical_sha256(report_payload)
    store = _store(database_url, schema)
    store.record_artifact(source_artifact, result.model_dump(mode="json"))
    probe_artifacts = (
        (target, "target-serving"),
        (dflash, "dflash-serving"),
        (capture, "target-hidden-state-capture"),
        (ingest, "specforge-batch-ingest"),
        (captured_optimizer, "captured-optimizer"),
        (candidate_serving, "candidate-speculative-serving"),
        (parent_restore, "parent-restore-and-cleanup"),
        (tokenizer_identity, "tokenizer-template-identity"),
        (sampled_distribution, "sampled-distribution-equivalence"),
        (training_model_artifact, "training-model-compatibility"),
    )
    for item, kind in probe_artifacts:
        if item is not None:
            store.record_artifact(item, {"kind": kind})
    store.record_compatibility_report("draft", draft_hash, report_payload, report_hash)
    _json(
        {
            "draft_hash": draft_hash,
            "upstream_incompatibility_verified": (result.upstream_incompatibility_verified),
            "ported_patch_applies": result.ported_patch.applies_cleanly,
            "training_model_compatible": training_model_port.upstream.compatible,
            "training_model_port_ready": training_model_port.ready_for_physical_probe,
            "report_hash": report_hash,
            "status": "blocked",
        }
    )
    raise typer.Exit(2)


def _build_source_compatibility_steps(
    source_artifact: ArtifactRef,
    target: ArtifactRef | None,
    dflash: ArtifactRef | None,
    capture: ArtifactRef | None,
    ingest: ArtifactRef | None,
    *,
    training_model: ArtifactRef | None = None,
    captured_optimizer: ArtifactRef | None = None,
    candidate_serving: ArtifactRef | None = None,
    parent_restore: ArtifactRef | None = None,
    tokenizer_identity: ArtifactRef | None = None,
    sampled_distribution: ArtifactRef | None = None,
    training_model_compatible: bool | None = None,
    training_model_port_ready: bool = False,
) -> tuple[CompatibilityStepResult, ...]:
    steps: list[CompatibilityStepResult] = []
    for name in COMPATIBILITY_LADDER:
        status = CompatibilityStatus.NOT_RUN
        evidence: tuple[ArtifactRef, ...] = ()
        evidence_level = EvidenceLevel.REAL_PROCESS
        detail = "not run after capture compatibility stop"
        if name in {"target-load", "target-only-serving"} and target is not None:
            status, evidence, detail = CompatibilityStatus.PASSED, (target,), "static target probe"
        elif name == "tokenizer-template-identity" and tokenizer_identity is not None:
            status = CompatibilityStatus.PASSED
            evidence = (tokenizer_identity,)
            evidence_level = EvidenceLevel.PHYSICAL_GPU
            detail = (
                "pinned target tokenizer files, loaded chat template, rendered prompt, "
                "vocabulary, and draft-worker target-tokenizer routing all match"
            )
        elif name == "official-dflash-load" and dflash is not None:
            status, evidence, detail = CompatibilityStatus.PASSED, (dflash,), "static DFlash probe"
        elif name == "greedy-lossless-parity" and target is not None and dflash is not None:
            status = CompatibilityStatus.PASSED
            evidence = (target, dflash)
            detail = "one request, fixed server seed, exact token and text parity"
        elif (
            name == "sampled-distribution-equivalence"
            and sampled_distribution is not None
        ):
            status = CompatibilityStatus.PASSED
            evidence = (sampled_distribution,)
            evidence_level = EvidenceLevel.PHYSICAL_GPU
            detail = (
                "power-calibrated sampled target/candidate equivalence stayed inside "
                "the signed target-self margin across fresh server starts"
            )
        elif name == "target-hidden-state-capture":
            if capture is not None:
                status = CompatibilityStatus.PASSED
                evidence = (source_artifact, capture)
                evidence_level = EvidenceLevel.PHYSICAL_GPU
                detail = (
                    "physical Laguna capture wrote five official DFlash layers "
                    "to Mooncake and cleaned all resources"
                )
            else:
                status = CompatibilityStatus.NOT_RUN
                evidence = (source_artifact,)
                detail = (
                    "ported capture patch applies to exact SGLang source; "
                    "physical Laguna capture has not passed"
                )
        elif name == "specforge-batch-ingest" and ingest is not None:
            status = CompatibilityStatus.PASSED
            evidence = (ingest,)
            evidence_level = EvidenceLevel.PHYSICAL_GPU
            detail = (
                "SpecForge materialized exact DFlash tensors, drained the release "
                "queue, bounded helper shutdown, and cleaned all resources"
            )
        elif (
            name
            in {
                "bounded-optimizer-step",
                "complete-checkpoint-save",
                "fresh-process-reload",
                "reloaded-logit-parity",
            }
            and captured_optimizer is not None
        ):
            status = CompatibilityStatus.PASSED
            evidence = (captured_optimizer,)
            evidence_level = EvidenceLevel.PHYSICAL_GPU
            details = {
                "bounded-optimizer-step": (
                    "live captured batch produced finite DFlash loss, required "
                    "Laguna gradients, and one AdamW parameter update"
                ),
                "complete-checkpoint-save": (
                    "content-addressed weights, optimizer, random state, cursor, "
                    "configuration, and manifest were written and committed"
                ),
                "fresh-process-reload": (
                    "a fresh H100 process loaded all 58 tensors with matching state, "
                    "buffer, layout, mode, and runtime configuration hashes"
                ),
                "reloaded-logit-parity": (
                    "the trained process and fresh reload produced bit-exact draft "
                    "outputs for identical hashed inputs"
                ),
            }
            detail = details[name]
        elif name == "candidate-speculative-serving" and candidate_serving is not None:
            status = CompatibilityStatus.PASSED
            evidence = (candidate_serving,)
            evidence_level = EvidenceLevel.PHYSICAL_GPU
            detail = (
                "trained checkpoint loaded in SGLang, performed nonzero DFlash "
                "proposal and verification work, preserved one greedy target output, "
                "and cleaned both server arms"
            )
        elif name in {"parent-restore", "resource-cleanup"} and parent_restore is not None:
            status = CompatibilityStatus.PASSED
            evidence = (parent_restore,)
            evidence_level = EvidenceLevel.PHYSICAL_GPU
            detail = (
                "fresh H100 restored the pinned official DFlash parent, reproduced "
                "the exact candidate request output with real speculative work, and "
                "left no process, GPU job, or open port"
            )
        elif name == "bounded-optimizer-step" and training_model is not None:
            evidence = (training_model,)
            if training_model_compatible is False and training_model_port_ready:
                detail = (
                    "the exact Laguna port applies and accounts for all checkpoint "
                    "tensors; physical model load and optimizer step have not run"
                )
            elif training_model_compatible is False:
                status = CompatibilityStatus.FAILED
                detail = (
                    "official Laguna DFlash uses DFlashLagunaForCausalLM, while "
                    "pinned SpecForge registers only its Qwen3-specific DFlashDraftModel"
                )
        steps.append(
            CompatibilityStepResult(
                name=name,
                status=status,
                evidence_level=evidence_level,
                evidence=evidence,
                detail=detail,
            )
        )
    return tuple(steps)


def _probe_artifact(path: Path) -> ArtifactRef | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact = payload.get("artifact")
    return ArtifactRef.model_validate(artifact) if isinstance(artifact, dict) else None


def _passing_probe_artifact(path: Path, expected_probe: str) -> ArtifactRef | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result")
    cleanup_passed = isinstance(result, dict) and (
        result.get("cleanup_passed") is True
        or (
            expected_probe == "candidate-serving"
            and isinstance(result.get("target_arm"), dict)
            and result["target_arm"].get("cleanup_passed") is True
            and isinstance(result.get("candidate_arm"), dict)
            and result["candidate_arm"].get("cleanup_passed") is True
        )
    )
    if (
        payload.get("probe") != expected_probe
        or payload.get("status") != "passed"
        or not isinstance(result, dict)
        or result.get("status") != "passed"
        or not cleanup_passed
    ):
        return None
    artifact = payload.get("artifact")
    return ArtifactRef.model_validate(artifact) if isinstance(artifact, dict) else None


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
    private_key: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    database_url: Annotated[str, typer.Option()] = os.environ.get(
        "AURORAPP_DATABASE_URL", "postgresql:///postgres"
    ),
    schema: Annotated[str, typer.Option()] = "aurorapp",
) -> None:
    value = HumanDecision.model_validate_json(decision.read_text(encoding="utf-8"))
    store = _store(database_url, schema)
    queued = store.review(review_id)
    if queued is None:
        raise typer.BadParameter(f"review {review_id} does not exist")
    question = queued["question"]
    if not isinstance(question, dict) or value.question != question.get("question"):
        raise typer.BadParameter("decision question does not match queued review")
    evidence = question.get("evidence")
    expected_hash = evidence.get("content_hash") if isinstance(evidence, dict) else None
    if expected_hash not in value.evidence_hashes:
        raise typer.BadParameter("decision does not include queued evidence hash")
    ApprovalSigner(private_key, value.reviewer).verify(value.approval_payload(), value.approval)
    store.answer_review(review_id, value.model_dump(mode="json"))
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
