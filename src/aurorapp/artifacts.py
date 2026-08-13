import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

from aurorapp.canonical import file_sha256
from aurorapp.models import ArtifactRef


class ArtifactError(RuntimeError):
    pass


class StagedArtifact:
    def __init__(self, path: Path, content_hash: str, size: int, producer: str) -> None:
        self.path = path
        self.content_hash = content_hash
        self.size = size
        self.producer = producer


class ContentAddressedArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.objects = root / "objects"
        self.staging = root / ".staging"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)

    def stage_bytes(self, name: str, value: bytes, producer: str) -> StagedArtifact:
        content_hash = hashlib.sha256(value).hexdigest()
        if self._object_directory(content_hash).exists():
            raise ArtifactError(f"artifact {content_hash} already exists")
        stage_directory = self.staging / str(uuid.uuid4())
        stage_directory.mkdir()
        path = stage_directory / name
        with path.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_directory(stage_directory)
        return StagedArtifact(path, content_hash, len(value), producer)

    def commit(
        self,
        staged: StagedArtifact,
        loader: Callable[[Path], bool],
    ) -> ArtifactRef:
        if file_sha256(staged.path) != staged.content_hash:
            self._discard(staged)
            raise ArtifactError("staged artifact hash changed")
        if not loader(staged.path):
            self._discard(staged)
            raise ArtifactError("loader rejected staged artifact")

        final_directory = self._object_directory(staged.content_hash)
        if final_directory.exists():
            self._discard(staged)
            raise ArtifactError(f"artifact {staged.content_hash} already exists")
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = f".{final_directory.name}.{uuid.uuid4()}.tmp"
        temporary_directory = final_directory.with_name(temporary_name)
        staged.path.parent.replace(temporary_directory)
        final_path = temporary_directory / staged.path.name
        manifest = {
            "content_hash": staged.content_hash,
            "size": staged.size,
            "producer": staged.producer,
            "file": staged.path.name,
            "validation_result": "valid",
        }
        manifest_path = temporary_directory / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._fsync_directory(temporary_directory)
        os.replace(temporary_directory, final_directory)
        self._fsync_directory(final_directory.parent)
        if file_sha256(final_directory / staged.path.name) != staged.content_hash:
            raise ArtifactError("published artifact failed final hash verification")
        return ArtifactRef(
            content_hash=staged.content_hash,
            size=staged.size,
            storage_path=str(final_directory / final_path.name),
            producer=staged.producer,
            validation_result="valid",
        )

    def resolve(self, content_hash: str) -> Path | None:
        directory = self._object_directory(content_hash)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text())
        filename = manifest.get("file")
        if not isinstance(filename, str):
            raise ArtifactError(f"artifact {content_hash} has an invalid manifest")
        path = directory / filename
        if not path.is_file() or file_sha256(path) != content_hash:
            raise ArtifactError(f"artifact {content_hash} is corrupt")
        return path

    def _object_directory(self, content_hash: str) -> Path:
        return self.objects / content_hash[:2] / content_hash

    def _discard(self, staged: StagedArtifact) -> None:
        shutil.rmtree(staged.path.parent, ignore_errors=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
