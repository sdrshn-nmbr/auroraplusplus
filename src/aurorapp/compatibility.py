import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from huggingface_hub import HfApi, hf_hub_download
from pydantic import computed_field, model_validator

from aurorapp.canonical import file_sha256
from aurorapp.models import ArtifactRef, EvidenceLevel, GitRevision, ModelRef, Sha256, StrictModel

COMPATIBILITY_LADDER = (
    "target-load",
    "tokenizer-template-identity",
    "target-only-serving",
    "official-dflash-load",
    "greedy-lossless-parity",
    "sampled-rng-contract",
    "target-hidden-state-capture",
    "specforge-batch-ingest",
    "bounded-optimizer-step",
    "complete-checkpoint-save",
    "fresh-process-reload",
    "reloaded-logit-parity",
    "candidate-speculative-serving",
    "parent-restore",
    "resource-cleanup",
)


class CompatibilityStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_RUN = "not-run"


class CompatibilityStepResult(StrictModel):
    name: str
    status: CompatibilityStatus
    evidence_level: EvidenceLevel
    evidence: tuple[ArtifactRef, ...]
    detail: str


class CompatibilityReport(StrictModel):
    identity_kind: Literal["draft", "run"]
    experiment_identity: Sha256
    steps: tuple[CompatibilityStepResult, ...]
    cleanup_verified: bool

    @model_validator(mode="after")
    def ladder_is_complete_and_ordered(self) -> "CompatibilityReport":
        if tuple(step.name for step in self.steps) != COMPATIBILITY_LADDER:
            raise ValueError("compatibility report must contain the exact ordered ladder")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def structurally_complete(self) -> bool:
        return self.cleanup_verified and all(
            step.status is CompatibilityStatus.PASSED
            and step.evidence_level is EvidenceLevel.PHYSICAL_GPU
            and step.evidence
            for step in self.steps
        )


class ArtifactResolver(Protocol):
    def resolve(self, content_hash: str) -> Path | None: ...


def verify_compatibility_report(
    report: CompatibilityReport,
    artifact_resolver: ArtifactResolver,
) -> bool:
    if report.identity_kind != "run" or not report.structurally_complete:
        return False
    expected_producer = f"experiment:{report.experiment_identity}:compatibility"
    for step in report.steps:
        for artifact in step.evidence:
            if artifact.validation_result != "valid" or artifact.producer != expected_producer:
                return False
            resolved = artifact_resolver.resolve(artifact.content_hash)
            if resolved is None or resolved.stat().st_size != artifact.size:
                return False
    return True


class ProbeParityResult(StrictModel):
    passed: bool
    mismatches: tuple[str, ...]


def verify_greedy_probe_parity(
    target: dict[str, object],
    challenger: dict[str, object],
) -> ProbeParityResult:
    mismatches: list[str] = []
    for identity in ("repository_revision", "modal_image_id"):
        if target.get(identity) != challenger.get(identity):
            mismatches.append(identity)
    target_result = target.get("result")
    challenger_result = challenger.get("result")
    if not isinstance(target_result, dict) or not isinstance(challenger_result, dict):
        return ProbeParityResult(passed=False, mismatches=("result",))
    if target_result.get("runtime") != challenger_result.get("runtime"):
        mismatches.append("runtime")
    target_response = _probe_response(target_result)
    challenger_response = _probe_response(challenger_result)
    for surface in ("output_ids", "text"):
        if target_response.get(surface) != challenger_response.get(surface):
            mismatches.append(surface)
    return ProbeParityResult(passed=not mismatches, mismatches=tuple(mismatches))


def _probe_response(result: dict[str, object]) -> dict[str, object]:
    generation = result.get("generation")
    if not isinstance(generation, dict):
        return {}
    response = generation.get("response_body")
    return response if isinstance(response, dict) else {}


class ModelManifestResolver:
    def __init__(self, cache_directory: Path) -> None:
        self.cache_directory = cache_directory
        self.api = HfApi()

    def resolve(
        self,
        repository: str,
        revision: GitRevision,
        precision: str,
        tokenizer_repository: str,
        tokenizer_revision: GitRevision,
    ) -> ModelRef:
        info = self.api.model_info(repository, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise RuntimeError(f"resolved model revision {info.sha} does not match {revision}")
        license_name = info.card_data.license if info.card_data is not None else None
        if not isinstance(license_name, str):
            raise RuntimeError(f"model {repository} has no machine-readable license")
        file_hashes: dict[str, str] = {}
        if info.siblings is None:
            raise RuntimeError(f"model {repository} returned no file manifest")
        for sibling in info.siblings:
            if sibling.lfs is not None:
                file_hashes[sibling.rfilename] = sibling.lfs.sha256
                continue
            path = Path(
                hf_hub_download(
                    repository,
                    sibling.rfilename,
                    revision=revision,
                    cache_dir=self.cache_directory,
                )
            )
            file_hashes[sibling.rfilename] = file_sha256(path)

        tokenizer_path = Path(
            hf_hub_download(
                tokenizer_repository,
                "tokenizer.json",
                revision=tokenizer_revision,
                cache_dir=self.cache_directory,
            )
        )
        template_path = Path(
            hf_hub_download(
                tokenizer_repository,
                "chat_template.jinja",
                revision=tokenizer_revision,
                cache_dir=self.cache_directory,
            )
        )
        return ModelRef(
            repository=repository,
            revision=revision,
            file_hashes=file_hashes,
            tokenizer_hash=file_sha256(tokenizer_path),
            chat_template_hash=file_sha256(template_path),
            precision=precision,
            license=license_name,
        )


def evidence_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
