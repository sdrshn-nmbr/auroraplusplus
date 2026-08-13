import hashlib
from pathlib import Path

from aurorapp.training_probe import inspect_training_model_compatibility


def test_official_laguna_checkpoint_is_rejected_by_qwen3_only_trainer(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    provider = tmp_path / "providers.py"
    model = tmp_path / "dflash.py"
    config.write_text(
        '{"architectures":["DFlashLagunaForCausalLM"],"model_type":"laguna"}',
        encoding="utf-8",
    )
    provider.write_text('DRAFT_ARCHITECTURE = "DFlashDraftModel"\n', encoding="utf-8")
    model.write_text(
        "class DFlashDraftModel(Qwen3PreTrainedModel):\n    config_class = Qwen3Config\n",
        encoding="utf-8",
    )

    result = inspect_training_model_compatibility(
        config,
        provider,
        model,
        aurorapp_revision="a" * 40,
        specforge_revision="b" * 40,
        specforge_main_revision="b" * 40,
    )

    assert result.compatible is False
    assert result.checkpoint_architecture == "DFlashLagunaForCausalLM"
    assert result.checkpoint_model_type == "laguna"
    assert result.trainer_architecture == "DFlashDraftModel"
    assert result.trainer_base_class == "Qwen3PreTrainedModel"
    assert result.trainer_config_class == "Qwen3Config"
    assert result.drafter_config_hash == hashlib.sha256(config.read_bytes()).hexdigest()
    assert "checkpoint architecture is not registered" in result.reasons
    assert "trainer implementation is Qwen3-specific" in result.reasons
