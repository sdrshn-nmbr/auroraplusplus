from pathlib import Path

import pytest

from aurorapp.controller import CommandConflictError, Controller, RecoveryCertainty
from aurorapp.models import (
    ActivationAuthorization,
    AuthorizedBootstrap,
    AuthorizedPromotion,
    EventType,
    EvidenceLevel,
    SystemMode,
)
from aurorapp.signatures import ApprovalSigner, VerifiedApproval
from aurorapp.simulation import FakeBoundaries, MemoryEventStore, VirtualClock

EXPERIMENT = "e" * 64
OFFICIAL = "0" * 64
PARENT = "1" * 64
CANDIDATE = "2" * 64


def make_controller(store: MemoryEventStore | None = None) -> Controller:
    return Controller(
        experiment_id=EXPERIMENT,
        event_store=store or MemoryEventStore(),
        boundaries=FakeBoundaries(),
        clock=VirtualClock(),
    )


def signed_bootstrap(tmp_path: Path) -> tuple[AuthorizedBootstrap, VerifiedApproval]:
    payload = {
        "experiment_id": EXPERIMENT,
        "drafter_id": PARENT,
        "parent_id": OFFICIAL,
        "artifact_manifest_hash": "3" * 64,
        "activation_hash": "6" * 64,
        "compatibility_report_hash": "7" * 64,
        "loader_verified": True,
    }
    signer = ApprovalSigner.generate(tmp_path / "bootstrap-key", identity="reviewer")
    approval = signer.sign(payload)
    return AuthorizedBootstrap(**payload, approval=approval), signer.verify(payload, approval)


def signed_activation(tmp_path: Path) -> tuple[ActivationAuthorization, VerifiedApproval]:
    payload = {
        "experiment_id": EXPERIMENT,
        "compatibility_report_hash": "7" * 64,
        "runtime_identity_hash": "8" * 64,
        "data_manifest_hash": "9" * 64,
        "evaluator_bundle_hash": "a" * 64,
        "repository_revision": "b" * 40,
        "compatibility_eligible": True,
        "artifacts_verified": True,
    }
    signer = ApprovalSigner.generate(tmp_path / "activation-key", identity="reviewer")
    approval = signer.sign(payload)
    return ActivationAuthorization(**payload, approval=approval), signer.verify(payload, approval)


def signed_promotion(tmp_path: Path) -> tuple[AuthorizedPromotion, VerifiedApproval]:
    payload = {
        "candidate_id": CANDIDATE,
        "parent_id": PARENT,
        "experiment_id": EXPERIMENT,
        "evaluator_bundle_hash": "4" * 64,
        "physical_report_hash": "5" * 64,
        "artifacts_verified": True,
        "loader_verified": True,
        "evidence_level": EvidenceLevel.PHYSICAL_GPU,
        "all_oracles_passed": True,
        "paired_gate_eligible": True,
    }
    signer = ApprovalSigner.generate(tmp_path / "promotion-key", identity="reviewer")
    approval = signer.sign(payload)
    return AuthorizedPromotion(**payload, approval=approval), signer.verify(payload, approval)


def test_uncertain_restart_fails_closed_to_shadow(tmp_path: Path) -> None:
    controller = make_controller()
    activation, approval = signed_activation(tmp_path)
    controller.enable(SystemMode.AUTO_DRAFTER, activation, approval, idempotency_key="enable")
    controller.recover(RecoveryCertainty.UNCERTAIN, idempotency_key="recover")

    assert controller.state.mode is SystemMode.SHADOW
    assert controller.events[-1].type is EventType.RECOVERED_TO_SHADOW


def test_duplicate_command_is_applied_once(tmp_path: Path) -> None:
    controller = make_controller()
    activation, approval = signed_activation(tmp_path)
    first = controller.enable(SystemMode.SHADOW, activation, approval, idempotency_key="same")
    second = controller.enable(SystemMode.SHADOW, activation, approval, idempotency_key="same")

    assert first == second
    assert len(controller.events) == 1


def test_conflicting_idempotency_content_is_rejected(tmp_path: Path) -> None:
    controller = make_controller()
    activation, approval = signed_activation(tmp_path)
    controller.enable(SystemMode.SHADOW, activation, approval, idempotency_key="same")

    with pytest.raises(CommandConflictError):
        controller.enable(SystemMode.AUTO_DRAFTER, activation, approval, idempotency_key="same")


def test_enable_requires_signed_activation(tmp_path: Path) -> None:
    controller = make_controller()
    activation, _approval = signed_activation(tmp_path)
    wrong_payload = activation.approval_payload() | {"runtime_identity_hash": "c" * 64}
    wrong_signer = ApprovalSigner.generate(tmp_path / "wrong-key", identity="other")
    wrong = wrong_signer.sign(wrong_payload)

    with pytest.raises(ValueError, match="does not match"):
        controller.enable(
            SystemMode.SHADOW,
            activation,
            VerifiedApproval(envelope=wrong),
            idempotency_key="enable",
        )


def test_stale_fence_cannot_promote(tmp_path: Path) -> None:
    controller = make_controller()
    bootstrap, bootstrap_approval = signed_bootstrap(tmp_path)
    controller.bootstrap(bootstrap, bootstrap_approval, idempotency_key="seed")
    activation, approval = signed_activation(tmp_path)
    controller.enable(SystemMode.AUTO_DRAFTER, activation, approval, idempotency_key="enable")
    controller.recover(RecoveryCertainty.CERTAIN, idempotency_key="ready")
    current = controller.acquire_fence("writer-a", "fence-a")
    controller.acquire_fence("writer-b", "fence-b")
    promotion, promotion_approval = signed_promotion(tmp_path)

    result = controller.promote(
        authorization=promotion,
        verified_approval=promotion_approval,
        fence=current,
        idempotency_key="promotion",
    )

    assert result.accepted is False
    assert controller.state.active_drafter == PARENT


def test_failed_learning_keeps_last_good_drafter(tmp_path: Path) -> None:
    controller = make_controller()
    bootstrap, approval = signed_bootstrap(tmp_path)
    controller.bootstrap(bootstrap, approval, idempotency_key="seed")
    controller.fail_learning("trainer OOM", idempotency_key="oom")

    assert controller.state.mode is SystemMode.FAILED
    assert controller.state.active_drafter == PARENT
    assert controller.boundaries.serving.target_only is False


def test_drain_cleans_every_owned_resource(tmp_path: Path) -> None:
    controller = make_controller()
    activation, approval = signed_activation(tmp_path)
    controller.enable(SystemMode.SHADOW, activation, approval, idempotency_key="enable")
    controller.boundaries.start_owned_resources()
    controller.drain(idempotency_key="drain")

    assert controller.state.mode is SystemMode.OFF
    assert controller.boundaries.live_resources == []


def test_restart_reconstructs_identical_state_from_events(tmp_path: Path) -> None:
    store = MemoryEventStore()
    original = make_controller(store)
    bootstrap, approval = signed_bootstrap(tmp_path)
    original.bootstrap(bootstrap, approval, idempotency_key="seed")
    activation, activation_approval = signed_activation(tmp_path)
    original.enable(
        SystemMode.AUTO_DRAFTER,
        activation,
        activation_approval,
        idempotency_key="enable",
    )
    original.recover(RecoveryCertainty.CERTAIN, idempotency_key="ready")
    original.acquire_fence("writer", "fence")

    restarted = make_controller(store)

    assert restarted.state == original.state
