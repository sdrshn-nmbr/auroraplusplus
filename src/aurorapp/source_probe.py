import hashlib
import subprocess
import tempfile
from pathlib import Path

from pydantic import Field

from aurorapp.models import GitRevision, Sha256, StrictModel

SGLANG_REPOSITORY = "https://github.com/sgl-project/sglang.git"
SGLANG_REVISION = "6a5a9eccaab0b98f5c3cfee170c7fbc87dfb98e5"
SPECFORGE_REPOSITORY = "https://github.com/sgl-project/SpecForge.git"
SPECFORGE_REVISION = "e6440f09a8574b35f894608559fd3d165971e488"
PATCH_PATH = "patches/sglang/v0.5.14/spec-capture.patch"
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


class SourceCompatibilityResult(StrictModel):
    aurorapp_revision: GitRevision
    sglang_revision: GitRevision
    specforge_revision: GitRevision
    patch_hash: Sha256
    command: tuple[str, ...]
    exit_code: int
    rejected_files: tuple[str, ...]
    expected_incompatibility_verified: bool
    diagnostic: str = Field(min_length=1)


def parse_rejected_files(diagnostic: str) -> tuple[str, ...]:
    suffix = ": patch does not apply"
    values = {
        line.removeprefix("error: ").removesuffix(suffix)
        for line in diagnostic.splitlines()
        if line.startswith("error: ") and line.endswith(suffix)
    }
    return tuple(sorted(values))


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
        patch = specforge / PATCH_PATH
        patch_hash = hashlib.sha256(patch.read_bytes()).hexdigest()
        command = ("git", "apply", "--check", str(patch))
        result = subprocess.run(
            command,
            cwd=sglang,
            check=False,
            capture_output=True,
            text=True,
        )
    diagnostic = result.stdout + result.stderr
    rejected = parse_rejected_files(diagnostic)
    expected = (
        result.returncode == 1
        and resolved_sglang == SGLANG_REVISION
        and resolved_specforge == SPECFORGE_REVISION
        and rejected == tuple(sorted(EXPECTED_REJECTED_FILES))
    )
    return SourceCompatibilityResult(
        aurorapp_revision=aurorapp_revision,
        sglang_revision=resolved_sglang,
        specforge_revision=resolved_specforge,
        patch_hash=patch_hash,
        command=command,
        exit_code=result.returncode,
        rejected_files=rejected,
        expected_incompatibility_verified=expected,
        diagnostic=diagnostic,
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
