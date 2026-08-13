from pathlib import Path

from aurorapp.artifacts import ContentAddressedArtifactStore
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
        experiment_run_id=RUN_ID,
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
        experiment_run_id=RUN_ID,
        steps=tuple(steps),
        cleanup_verified=True,
    )

    assert verify_compatibility_report(report, store) is False


def test_cleanup_failure_blocks_activation(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path)
    report = CompatibilityReport(
        experiment_run_id=RUN_ID,
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
        experiment_run_id=RUN_ID,
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
