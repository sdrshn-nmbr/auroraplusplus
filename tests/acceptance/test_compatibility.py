import json
from pathlib import Path

from aurorapp.artifacts import ContentAddressedArtifactStore
from aurorapp.cli import _build_source_compatibility_steps, _passing_probe_artifact
from aurorapp.compatibility import (
    COMPATIBILITY_LADDER,
    CompatibilityReport,
    CompatibilityStatus,
    CompatibilityStepResult,
    verify_compatibility_report,
    verify_greedy_probe_parity,
)
from aurorapp.models import ArtifactRef, EvidenceLevel

RUN_ID = "b" * 64


def evidence(store: ContentAddressedArtifactStore, name: str) -> ArtifactRef:
    staged = store.stage_bytes(name, name.encode(), producer=f"experiment:{RUN_ID}:compatibility")
    return store.commit(staged, loader=lambda path: path.read_bytes() == name.encode())


def result(
    name: str,
    status: CompatibilityStatus,
    item: ArtifactRef,
) -> CompatibilityStepResult:
    return CompatibilityStepResult(
        name=name,
        status=status,
        evidence_level=EvidenceLevel.PHYSICAL_GPU,
        evidence=(item,),
        detail="test",
    )


def test_report_requires_every_resolvable_step_in_exact_order(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path)
    steps = tuple(
        result(name, CompatibilityStatus.PASSED, evidence(store, name))
        for name in COMPATIBILITY_LADDER
    )
    report = CompatibilityReport(
        identity_kind="run",
        experiment_identity=RUN_ID,
        steps=steps,
        cleanup_verified=True,
    )

    assert verify_compatibility_report(report, store) is True


def test_one_not_run_step_blocks_activation(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path)
    steps = [
        result(name, CompatibilityStatus.PASSED, evidence(store, name))
        for name in COMPATIBILITY_LADDER
    ]
    steps[7] = result(
        COMPATIBILITY_LADDER[7],
        CompatibilityStatus.NOT_RUN,
        steps[7].evidence[0],
    )
    report = CompatibilityReport(
        identity_kind="run",
        experiment_identity=RUN_ID,
        steps=tuple(steps),
        cleanup_verified=True,
    )

    assert verify_compatibility_report(report, store) is False


def test_cleanup_failure_blocks_activation(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path)
    report = CompatibilityReport(
        identity_kind="run",
        experiment_identity=RUN_ID,
        steps=tuple(
            result(name, CompatibilityStatus.PASSED, evidence(store, name))
            for name in COMPATIBILITY_LADDER
        ),
        cleanup_verified=False,
    )

    assert verify_compatibility_report(report, store) is False


def test_unresolvable_or_wrong_runtime_evidence_blocks_activation(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path)
    refs = [evidence(store, name) for name in COMPATIBILITY_LADDER]
    wrong = refs[0].model_copy(update={"producer": "experiment:wrong:compatibility"})
    refs[0] = wrong
    report = CompatibilityReport(
        identity_kind="run",
        experiment_identity=RUN_ID,
        steps=tuple(
            result(name, CompatibilityStatus.PASSED, item)
            for name, item in zip(COMPATIBILITY_LADDER, refs, strict=True)
        ),
        cleanup_verified=True,
    )

    assert verify_compatibility_report(report, store) is False


def test_greedy_probe_parity_compares_exact_token_and_text_surfaces() -> None:
    target = {
        "repository_revision": RUN_ID,
        "modal_image_id": "image",
        "result": {
            "generation": {"response_body": {"output_ids": [1, 2], "text": "ok"}},
            "runtime": {"cuda_base_image": "cuda"},
        },
    }
    challenger = target | {
        "result": {
            "generation": {"response_body": {"output_ids": [1, 3], "text": "bad"}},
            "runtime": {"cuda_base_image": "cuda"},
        }
    }

    result = verify_greedy_probe_parity(target, challenger)

    assert result.passed is False
    assert result.mismatches == ("output_ids", "text")


def test_capture_artifact_requires_probe_result_and_cleanup_to_pass(tmp_path: Path) -> None:
    item = ArtifactRef(
        content_hash="a" * 64,
        producer="probe",
        size=1,
        storage_path="objects/a",
        validation_result="valid",
    )
    path = tmp_path / "capture.json"
    payload = {
        "probe": "capture",
        "status": "passed",
        "result": {"status": "passed", "cleanup_passed": True},
        "artifact": item.model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _passing_probe_artifact(path, "capture") == item

    payload["result"]["cleanup_passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _passing_probe_artifact(path, "capture") is None


def test_source_report_binds_passing_specforge_ingest_evidence() -> None:
    items = [
        ArtifactRef(
            content_hash=str(index) * 64,
            producer="probe",
            size=1,
            storage_path=f"objects/{index}",
            validation_result="valid",
        )
        for index in range(1, 7)
    ]

    steps = _build_source_compatibility_steps(
        *items[:5],
        training_model=items[5],
        training_model_compatible=False,
        training_model_port_ready=True,
    )
    ingest = steps[COMPATIBILITY_LADDER.index("specforge-batch-ingest")]

    assert ingest.status is CompatibilityStatus.PASSED
    assert ingest.evidence_level is EvidenceLevel.PHYSICAL_GPU
    assert ingest.evidence == (items[4],)
    optimizer = steps[COMPATIBILITY_LADDER.index("bounded-optimizer-step")]
    assert optimizer.status is CompatibilityStatus.NOT_RUN
    assert optimizer.evidence == (items[5],)
