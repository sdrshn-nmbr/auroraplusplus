from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

from aurorapp.models import GitRevision, Sha256, StrictModel

SGLANG_REPOSITORY = "https://github.com/sgl-project/sglang.git"
SGLANG_REVISION = "6a5a9eccaab0b98f5c3cfee170c7fbc87dfb98e5"
SPECFORGE_REPOSITORY = "https://github.com/sgl-project/SpecForge.git"
SPECFORGE_REVISION = "e6440f09a8574b35f894608559fd3d165971e488"
PATCH_PATH = "patches/sglang/v0.5.14/spec-capture.patch"
PORT_PATCH_PATH = Path("patches/sglang/6a5a9ec/spec-capture.patch")
EXPECTED_REJECTED_FILES = (
    "python/sglang/srt/layers/logits_processor.py",
    "python/sglang/srt/managers/detokenizer_manager.py",
    "python/sglang/srt/managers/io_struct.py",
    "python/sglang/srt/managers/schedule_batch.py",
    "python/sglang/srt/managers/scheduler.py",
    "python/sglang/srt/managers/scheduler_components/batch_result_processor.py",
    "python/sglang/srt/managers/scheduler_components/output_streamer.py",
    "python/sglang/srt/managers/utils.py",
    "python/sglang/srt/model_executor/model_runner.py",
    "python/sglang/srt/server_args.py",
)


class PatchApplicationResult(StrictModel):
    repository_revision: GitRevision
    patch_hash: Sha256
    command: tuple[str, ...]
    exit_code: int
    rejected_files: tuple[str, ...]
    applies_cleanly: bool
    diagnostic: str


class SourceCompatibilityResult(StrictModel):
    aurorapp_revision: GitRevision
    sglang_revision: GitRevision
    specforge_revision: GitRevision
    upstream_patch: PatchApplicationResult
    ported_patch: PatchApplicationResult
    upstream_incompatibility_verified: bool


def parse_rejected_files(diagnostic: str) -> tuple[str, ...]:
    suffix = ": patch does not apply"
    values = {
        line.removeprefix("error: ").removesuffix(suffix)
        for line in diagnostic.splitlines()
        if line.startswith("error: ") and line.endswith(suffix)
    }
    return tuple(sorted(values))


def check_patch_application(
    repository: Path,
    expected_revision: str,
    patch: Path,
) -> PatchApplicationResult:
    repository_revision = _revision(repository)
    if repository_revision != expected_revision:
        raise ValueError(
            "source tree revision mismatch: "
            f"expected {expected_revision}, got {repository_revision}"
        )
    patch_hash = hashlib.sha256(patch.read_bytes()).hexdigest()
    command = ("git", "apply", "--check", str(patch))
    completed = subprocess.run(
        command,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    diagnostic = completed.stdout + completed.stderr
    rejected_files = parse_rejected_files(diagnostic)
    return PatchApplicationResult(
        repository_revision=repository_revision,
        patch_hash=patch_hash,
        command=command,
        exit_code=completed.returncode,
        rejected_files=rejected_files,
        applies_cleanly=completed.returncode == 0,
        diagnostic=diagnostic,
    )


def run_capture_patch_probe(aurorapp_repository: Path) -> SourceCompatibilityResult:
    aurorapp_revision = _revision(aurorapp_repository)
    with tempfile.TemporaryDirectory(prefix="aurorapp-source-probe-") as temporary:
        root = Path(temporary)
        sglang = root / "sglang"
        specforge = root / "specforge"
        _clone_exact(SGLANG_REPOSITORY, SGLANG_REVISION, sglang)
        _clone_exact(SPECFORGE_REPOSITORY, SPECFORGE_REVISION, specforge)
        resolved_sglang = _revision(sglang)
        resolved_specforge = _revision(specforge)
        upstream = check_patch_application(
            sglang,
            resolved_sglang,
            specforge / PATCH_PATH,
        )
        ported = check_patch_application(
            sglang,
            resolved_sglang,
            aurorapp_repository / PORT_PATCH_PATH,
        )
    expected = (
        upstream.exit_code == 1
        and resolved_sglang == SGLANG_REVISION
        and resolved_specforge == SPECFORGE_REVISION
        and upstream.rejected_files == tuple(sorted(EXPECTED_REJECTED_FILES))
    )
    return SourceCompatibilityResult(
        aurorapp_revision=aurorapp_revision,
        sglang_revision=resolved_sglang,
        specforge_revision=resolved_specforge,
        upstream_patch=upstream,
        ported_patch=ported,
        upstream_incompatibility_verified=expected,
    )


def _clone_exact(repository: str, revision: str, destination: Path) -> None:
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "remote", "add", "origin", repository], check=True
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "-q", "--depth", "1", "origin", revision],
        check=True,
    )
    subprocess.run(["git", "-C", str(destination), "checkout", "-q", "FETCH_HEAD"], check=True)


def _revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
