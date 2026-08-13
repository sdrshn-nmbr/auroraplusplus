from pathlib import Path

import pytest

from aurorapp.signatures import ApprovalError, ApprovalSigner


def test_ssh_approval_covers_exact_canonical_payload(tmp_path: Path) -> None:
    key = tmp_path / "approval"
    signer = ApprovalSigner.generate(key, identity="test-reviewer")
    payload = {"bundle": "physical-v1", "threshold": 0.03}

    approval = signer.sign(payload)

    signer.verify(payload, approval)
    with pytest.raises(ApprovalError, match="payload hash"):
        signer.verify({"bundle": "physical-v1", "threshold": 0.01}, approval)
