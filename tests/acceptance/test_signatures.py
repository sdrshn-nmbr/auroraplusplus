from pathlib import Path

import pytest

from aurorapp.models import HumanDecision
from aurorapp.signatures import ApprovalError, ApprovalSigner


def test_ssh_approval_covers_exact_canonical_payload(tmp_path: Path) -> None:
    key = tmp_path / "approval"
    signer = ApprovalSigner.generate(key, identity="test-reviewer")
    payload = {"bundle": "physical-v1", "threshold": 0.03}

    approval = signer.sign(payload)

    signer.verify(payload, approval)
    with pytest.raises(ApprovalError, match="payload hash"):
        signer.verify({"bundle": "physical-v1", "threshold": 0.01}, approval)


def test_human_decision_signature_targets_exact_decision(tmp_path: Path) -> None:
    signer = ApprovalSigner.generate(tmp_path / "approval", identity="reviewer")
    payload = {
        "decision_id": "decision-1",
        "question": "choose",
        "answer": "stop",
        "reason": "capture incompatible",
        "evidence_hashes": ["a" * 64],
        "reviewer": "reviewer",
    }

    decision = HumanDecision(**payload, approval=signer.sign(payload))

    assert decision.approval.payload_hash
