from pathlib import Path

import pytest

from aurorapp.artifacts import ArtifactError, ContentAddressedArtifactStore


def test_artifact_is_invisible_until_hash_and_loader_pass(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path)
    staged = store.stage_bytes("weights.bin", b"candidate-weights", producer="test")

    assert store.resolve(staged.content_hash) is None
    with pytest.raises(ArtifactError, match="loader"):
        store.commit(staged, loader=lambda _: False)

    assert store.resolve(staged.content_hash) is None


def test_artifact_publish_is_immutable_and_loadable(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path)
    staged = store.stage_bytes("weights.bin", b"candidate-weights", producer="test")
    committed = store.commit(staged, loader=lambda path: path.read_bytes() == b"candidate-weights")

    assert store.resolve(committed.content_hash) == Path(committed.storage_path)
    assert committed.validation_result == "valid"
    with pytest.raises(ArtifactError, match="already exists"):
        store.stage_bytes("weights.bin", b"candidate-weights", producer="test")
