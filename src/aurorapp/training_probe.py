import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from pydantic import computed_field

from aurorapp.model_port import (
    CheckpointHeaderResult,
    LagunaDFlashConfigContract,
    read_safetensors_header,
    validate_checkpoint_header,
)
from aurorapp.models import GitRevision, Sha256, StrictModel
from aurorapp.source_probe import (
    SPECFORGE_REPOSITORY,
    SPECFORGE_REVISION,
    PatchApplicationResult,
    _clone_exact,
    _revision,
    check_patch_application,
)

DRAFTER_REPOSITORY = "poolside/Laguna-XS-2.1-DFlash-INT4"
DRAFTER_REVISION = "630267be6a2ad870ca80fc3930bde87d6ad7bb24"
PROVIDER_PATH = Path("specforge/algorithms/dflash/providers.py")
MODEL_PATH = Path("specforge/modeling/draft/dflash.py")
PORT_PATCH_PATH = Path("patches/specforge/e6440f09/laguna-dflash-training.patch")


class TrainingModelCompatibilityResult(StrictModel):
    aurorapp_revision: GitRevision
    specforge_revision: GitRevision
    specforge_main_revision: GitRevision
    drafter_repository: str
    drafter_revision: GitRevision
    drafter_config_hash: Sha256
    provider_hash: Sha256
    model_source_hash: Sha256
    checkpoint_architecture: str
    checkpoint_model_type: str
    trainer_architecture: str
    trainer_base_class: str
    trainer_config_class: str
    compatible: bool
    reasons: tuple[str, ...]


class TrainingModelPortProbeResult(StrictModel):
    upstream: TrainingModelCompatibilityResult
    ported_patch: PatchApplicationResult
    checkpoint_header: CheckpointHeaderResult

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ready_for_physical_probe(self) -> bool:
        return (
            not self.upstream.compatible
            and self.ported_patch.applies_cleanly
            and self.checkpoint_header.passed
        )


def inspect_training_model_compatibility(
    config_path: Path,
    provider_path: Path,
    model_path: Path,
    *,
    aurorapp_revision: str,
    specforge_revision: str,
    specforge_main_revision: str,
) -> TrainingModelCompatibilityResult:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("drafter config must declare exactly one architecture")
    checkpoint_architecture = architectures[0]
    checkpoint_model_type = config.get("model_type")
    if not isinstance(checkpoint_architecture, str) or not isinstance(checkpoint_model_type, str):
        raise ValueError("drafter config architecture and model type must be strings")

    provider_tree = ast.parse(provider_path.read_text(encoding="utf-8"))
    model_tree = ast.parse(model_path.read_text(encoding="utf-8"))
    trainer_architecture = _module_string(provider_tree, "DRAFT_ARCHITECTURE")
    trainer_class = _class(model_tree, trainer_architecture)
    if len(trainer_class.bases) != 1:
        raise ValueError("SpecForge DFlash trainer must declare exactly one base class")
    trainer_base_class = _expression_name(trainer_class.bases[0])
    trainer_config_class = _class_name_assignment(trainer_class, "config_class")

    reasons: list[str] = []
    if checkpoint_architecture != trainer_architecture:
        reasons.append("checkpoint architecture is not registered")
    if checkpoint_model_type != "qwen3" and (
        trainer_base_class.startswith("Qwen3") or trainer_config_class.startswith("Qwen3")
    ):
        reasons.append("trainer implementation is Qwen3-specific")
    return TrainingModelCompatibilityResult(
        aurorapp_revision=aurorapp_revision,
        specforge_revision=specforge_revision,
        specforge_main_revision=specforge_main_revision,
        drafter_repository=DRAFTER_REPOSITORY,
        drafter_revision=DRAFTER_REVISION,
        drafter_config_hash=_hash(config_path),
        provider_hash=_hash(provider_path),
        model_source_hash=_hash(model_path),
        checkpoint_architecture=checkpoint_architecture,
        checkpoint_model_type=checkpoint_model_type,
        trainer_architecture=trainer_architecture,
        trainer_base_class=trainer_base_class,
        trainer_config_class=trainer_config_class,
        compatible=not reasons,
        reasons=tuple(reasons),
    )


def run_training_model_probe(
    aurorapp_repository: Path,
) -> TrainingModelCompatibilityResult:
    config_path = Path(
        hf_hub_download(
            DRAFTER_REPOSITORY,
            "config.json",
            revision=DRAFTER_REVISION,
        )
    )
    with tempfile.TemporaryDirectory(prefix="aurorapp-training-probe-") as temporary:
        specforge = Path(temporary) / "specforge"
        _clone_exact(SPECFORGE_REPOSITORY, SPECFORGE_REVISION, specforge)
        return inspect_training_model_compatibility(
            config_path,
            specforge / PROVIDER_PATH,
            specforge / MODEL_PATH,
            aurorapp_revision=_revision(aurorapp_repository),
            specforge_revision=_revision(specforge),
            specforge_main_revision=_remote_main_revision(),
        )


def run_training_model_port_probe(
    aurorapp_repository: Path,
) -> TrainingModelPortProbeResult:
    upstream = run_training_model_probe(aurorapp_repository)
    config_path = Path(
        hf_hub_download(
            DRAFTER_REPOSITORY,
            "config.json",
            revision=DRAFTER_REVISION,
        )
    )
    weights_path = Path(
        hf_hub_download(
            DRAFTER_REPOSITORY,
            "model.safetensors",
            revision=DRAFTER_REVISION,
        )
    )
    config = LagunaDFlashConfigContract.model_validate_json(config_path.read_text(encoding="utf-8"))
    checkpoint_header = validate_checkpoint_header(
        config,
        read_safetensors_header(weights_path),
    )
    with tempfile.TemporaryDirectory(prefix="aurorapp-training-port-probe-") as temporary:
        specforge = Path(temporary) / "specforge"
        _clone_exact(SPECFORGE_REPOSITORY, SPECFORGE_REVISION, specforge)
        ported_patch = check_patch_application(
            specforge,
            SPECFORGE_REVISION,
            aurorapp_repository / PORT_PATCH_PATH,
        )
    return TrainingModelPortProbeResult(
        upstream=upstream,
        ported_patch=ported_patch,
        checkpoint_header=checkpoint_header,
    )


def _remote_main_revision() -> str:
    result = subprocess.run(
        ["git", "ls-remote", SPECFORGE_REPOSITORY, "refs/heads/main"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision, separator, reference = result.stdout.strip().partition("\t")
    if not separator or reference != "refs/heads/main":
        raise ValueError("SpecForge main did not resolve to one exact revision")
    return revision


def _module_string(tree: ast.Module, name: str) -> str:
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name for target in statement.targets
            )
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            return statement.value.value
    raise ValueError(f"missing string assignment {name}")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for statement in tree.body:
        if isinstance(statement, ast.ClassDef) and statement.name == name:
            return statement
    raise ValueError(f"missing class {name}")


def _class_name_assignment(node: ast.ClassDef, name: str) -> str:
    for statement in node.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in statement.targets
        ):
            return _expression_name(statement.value)
    raise ValueError(f"missing class assignment {name}")


def _expression_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_name(node.value)}.{node.attr}"
    raise ValueError("expected a named expression")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
