import base64
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aurorapp.canonical import canonical_bytes, canonical_sha256
from aurorapp.models import HumanApproval


class ApprovalError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedApproval:
    envelope: HumanApproval


class ApprovalSigner:
    namespace = "aurorapp"

    def __init__(self, private_key: Path, identity: str) -> None:
        self.private_key = private_key
        self.public_key_path = Path(f"{private_key}.pub")
        self.identity = identity
        if not private_key.is_file() or not self.public_key_path.is_file():
            raise ApprovalError(f"approval key pair does not exist at {private_key}")
        self.public_key = self.public_key_path.read_text(encoding="utf-8").strip()

    @classmethod
    def generate(cls, private_key: Path, identity: str) -> "ApprovalSigner":
        private_key.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                identity,
                "-f",
                str(private_key),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ApprovalError(f"ssh-keygen failed: {result.stderr.strip()}")
        return cls(private_key, identity)

    def sign(self, payload: Any) -> HumanApproval:
        value = canonical_bytes(payload)
        with tempfile.TemporaryDirectory(prefix="aurorapp-sign-") as temporary:
            payload_path = Path(temporary) / "payload.json"
            payload_path.write_bytes(value)
            result = subprocess.run(
                [
                    "ssh-keygen",
                    "-Y",
                    "sign",
                    "-f",
                    str(self.private_key),
                    "-n",
                    self.namespace,
                    str(payload_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise ApprovalError(f"ssh signing failed: {result.stderr.strip()}")
            signature = base64.b64encode(Path(f"{payload_path}.sig").read_bytes()).decode("ascii")
        approval = HumanApproval(
            payload_hash=canonical_sha256(payload),
            signature=signature,
            public_key=self.public_key,
            signer=self.identity,
            signed_at=datetime.now(UTC),
        )
        self.verify(payload, approval)
        return approval

    def verify(self, payload: Any, approval: HumanApproval) -> VerifiedApproval:
        if canonical_sha256(payload) != approval.payload_hash:
            raise ApprovalError("payload hash does not match approval")
        if approval.public_key != self.public_key:
            raise ApprovalError("approval public key is not trusted by this signer")
        value = canonical_bytes(payload)
        with tempfile.TemporaryDirectory(prefix="aurorapp-verify-") as temporary:
            root = Path(temporary)
            allowed_signers = root / "allowed_signers"
            signature_path = root / "payload.sig"
            allowed_signers.write_text(
                f"{approval.signer} {approval.public_key}\n", encoding="utf-8"
            )
            signature_path.write_bytes(base64.b64decode(approval.signature))
            result = subprocess.run(
                [
                    "ssh-keygen",
                    "-Y",
                    "verify",
                    "-f",
                    str(allowed_signers),
                    "-I",
                    approval.signer,
                    "-n",
                    self.namespace,
                    "-s",
                    str(signature_path),
                ],
                input=value,
                check=False,
                capture_output=True,
            )
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise ApprovalError(f"SSH signature verification failed: {detail}")
        return VerifiedApproval(envelope=approval)
