import json
from pathlib import Path

import pytest

from aurorapp.model_port import (
    CapturedBatchOptimizerResult,
    CheckpointReloadResult,
    DraftExecutionState,
    DraftProbeInputHashes,
    LagunaDFlashConfigContract,
    PhysicalModelPortResult,
    checkpoint_contract,
    draft_runtime_config,
    probe_wire_json,
    validate_checkpoint_header,
)


def execution_state() -> DraftExecutionState:
    return DraftExecutionState(
        nonpersistent_buffer_digest="a" * 64,
        parameter_layout_digest="b" * 64,
        module_modes_digest="c" * 64,
        runtime_config_digest="d" * 64,
        float32_matmul_precision="highest",
        cudnn_benchmark=False,
        cudnn_deterministic=False,
        deterministic_algorithms=False,
        cublas_allow_tf32=False,
        cudnn_allow_tf32=True,
    )


def test_training_wrapper_does_not_cast_nonpersistent_model_buffers() -> None:
    source = (
        Path(__file__).parents[2] / "scripts" / "specforge_captured_optimizer_probe.py"
    ).read_text(encoding="utf-8")

    assert 'loss_type="dflash",\n    ).to(device="cuda")' in source
    assert 'loss_type="dflash",\n    ).to(device="cuda", dtype=torch.bfloat16)' not in source


def test_runtime_config_ignores_load_path_but_keeps_execution_fields() -> None:
    base = {
        "_name_or_path": "poolside/draft",
        "model_type": "laguna",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 5,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
        "sliding_window": 512,
        "rope_theta": 500000.0,
        "attention_dropout": 0.0,
        "gating": "per-head",
        "layer_types": ["sliding_attention"] * 5,
        "dflash_config": {
            "block_size": 16,
            "mask_token_id": 12,
            "num_target_layers": 40,
            "target_layer_ids": [1, 13, 25, 33, 39],
            "causal": True,
        },
    }

    hub = draft_runtime_config(base, attention_implementation="eager")
    checkpoint = draft_runtime_config(
        {**base, "_name_or_path": "/checkpoints/candidate"},
        attention_implementation="eager",
    )

    assert hub == checkpoint
    assert draft_runtime_config(base, attention_implementation="sdpa") != hub


def official_config() -> LagunaDFlashConfigContract:
    return LagunaDFlashConfigContract.model_validate_json(
        json.dumps(
            {
                "architectures": ["DFlashLagunaForCausalLM"],
                "model_type": "laguna",
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 5,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "attention_bias": False,
                "gating": "per-head",
                "layer_types": ["sliding_attention"] * 5,
                "dflash_config": {
                    "block_size": 16,
                    "mask_token_id": 12,
                    "num_target_layers": 40,
                    "target_layer_ids": [1, 13, 25, 33, 39],
                    "causal": True,
                },
            }
        )
    )


def test_official_checkpoint_contract_accounts_for_every_laguna_tensor() -> None:
    contract = checkpoint_contract(official_config())

    assert len(contract) == 58
    assert contract["layers.0.self_attn.qkv_proj.weight"].shape == (10240, 2048)
    assert contract["layers.0.self_attn.g_proj.weight"].shape == (64, 2048)
    assert contract["aux_hidden_norms.4.weight"].shape == (2048,)
    assert contract["fc.weight"].shape == (2048, 10240)


def test_checkpoint_header_rejects_one_dropped_laguna_gate() -> None:
    expected = checkpoint_contract(official_config())
    header = {
        name: {"dtype": tensor.dtype, "shape": list(tensor.shape)}
        for name, tensor in expected.items()
    }
    header.pop("layers.3.self_attn.g_proj.weight")

    result = validate_checkpoint_header(official_config(), header)

    assert result.passed is False
    assert result.missing == ("layers.3.self_attn.g_proj.weight",)
    assert result.unexpected == ()
    assert result.shape_or_dtype_mismatches == ()


def test_physical_port_result_requires_every_laguna_gradient_surface() -> None:
    result = PhysicalModelPortResult.model_validate(
        {
            "architecture": "DFlashLagunaForCausalLM",
            "checkpoint": {
                "passed": True,
                "expected_count": 58,
                "observed_count": 58,
                "missing": [],
                "unexpected": [],
                "shape_or_dtype_mismatches": [],
            },
            "loading_missing": [],
            "loading_unexpected": [],
            "loading_mismatched": [],
            "forward_shape": [1, 4, 2048],
            "loss_finite": True,
            "gradient_parameters": [
                "layers.0.self_attn.qkv_proj.weight",
                "layers.0.self_attn.g_proj.weight",
                "fc.weight",
                "aux_hidden_norms.0.weight",
            ],
            "optimizer_state_entries": 58,
            "changed_parameter": "layers.0.self_attn.g_proj.weight",
            "parameter_delta": 1.0,
        }
    )

    assert result.passed is True

    payload = result.model_dump(mode="json", exclude={"passed"})
    payload["gradient_parameters"].remove("aux_hidden_norms.0.weight")
    invalid = PhysicalModelPortResult.model_validate(payload)
    assert invalid.passed is False


def test_captured_batch_optimizer_requires_reload_and_complete_checkpoint() -> None:
    payload = {
        "sample_id": "laguna-live-batch-1",
        "input_ids_shape": [1, 4],
        "loss_mask_shape": [1, 4],
        "hidden_states_shape": [1, 4, 10240],
        "loss": 1.25,
        "accuracy": 0.5,
        "accuracy_denom": 2,
        "gradient_parameters": [
            "draft_model.layers.0.self_attn.qkv_proj.weight",
            "draft_model.layers.0.self_attn.g_proj.weight",
            "draft_model.fc.weight",
            "draft_model.aux_hidden_norms.0.weight",
        ],
        "optimizer_state_entries": 58,
        "changed_parameter": "draft_model.layers.0.self_attn.g_proj.weight",
        "parameter_delta": 1.0,
        "checkpoint_hashes": {
            "config": "e" * 64,
            "weights": "a" * 64,
            "optimizer": "b" * 64,
            "random_state": "c" * 64,
            "manifest": "d" * 64,
        },
        "checkpoint_path": "/checkpoints/objects/dd/candidate",
        "pre_save_state_digest": "f" * 64,
        "training_cursor": 1,
        "reload": {
            "missing": [],
            "unexpected": [],
            "state_digest": "f" * 64,
            "reference_state_digest": "f" * 64,
            "state_equal": True,
            "input_hashes": {
                "noise_embedding": "1" * 64,
                "target_hidden": "2" * 64,
                "position_ids": "3" * 64,
            },
            "inputs_equal": True,
            "reference_execution_state": execution_state().model_dump(mode="json"),
            "observed_execution_state": execution_state().model_dump(mode="json"),
            "reference_repeat_equal": True,
            "observed_repeat_equal": True,
            "output_equal": True,
            "output_allclose": True,
            "output_mismatch_count": 0,
            "output_max_abs_difference": 0,
            "output_mean_abs_difference": 0,
        },
        "released": True,
        "release_pending": 0,
    }

    result = CapturedBatchOptimizerResult.model_validate(payload)

    assert result.passed is True
    payload["checkpoint_hashes"].pop("random_state")
    assert CapturedBatchOptimizerResult.model_validate(payload).passed is False


def test_captured_batch_optimizer_requires_checkpoint_path_at_construction() -> None:
    with pytest.raises(ValueError, match="checkpoint_path"):
        CapturedBatchOptimizerResult.model_validate(
            {
                "sample_id": "sample",
                "input_ids_shape": [1, 4],
                "loss_mask_shape": [1, 4],
                "hidden_states_shape": [1, 4, 10240],
                "loss": 1.0,
                "accuracy": 0.0,
                "accuracy_denom": 1,
                "gradient_parameters": [],
                "optimizer_state_entries": 0,
                "changed_parameter": "draft_model.layers.0.self_attn.g_proj.weight",
                "parameter_delta": 0.0,
                "checkpoint_hashes": {},
                "pre_save_state_digest": "f" * 64,
                "training_cursor": 1,
                "reload": {
                    "missing": [],
                    "unexpected": [],
                    "state_digest": "f" * 64,
                    "reference_state_digest": "f" * 64,
                    "state_equal": True,
                    "input_hashes": {
                        "noise_embedding": "1" * 64,
                        "target_hidden": "2" * 64,
                        "position_ids": "3" * 64,
                    },
                    "inputs_equal": True,
                    "reference_execution_state": execution_state().model_dump(mode="json"),
                    "observed_execution_state": execution_state().model_dump(mode="json"),
                    "reference_repeat_equal": True,
                    "observed_repeat_equal": True,
                    "output_equal": True,
                    "output_allclose": True,
                    "output_mismatch_count": 0,
                    "output_max_abs_difference": 0,
                    "output_mean_abs_difference": 0,
                },
                "released": True,
                "release_pending": 0,
            }
        )


def test_reload_diagnostic_does_not_replace_exact_parity_with_allclose() -> None:
    result = CheckpointReloadResult(
        missing=(),
        unexpected=(),
        state_digest="a" * 64,
        reference_state_digest="a" * 64,
        state_equal=True,
        input_hashes=DraftProbeInputHashes(
            noise_embedding="1" * 64,
            target_hidden="2" * 64,
            position_ids="3" * 64,
        ),
        inputs_equal=True,
        reference_execution_state=execution_state(),
        observed_execution_state=execution_state(),
        reference_repeat_equal=True,
        observed_repeat_equal=True,
        output_equal=False,
        output_allclose=True,
        output_mismatch_count=1,
        output_max_abs_difference=0.001,
        output_mean_abs_difference=0.0001,
    )

    assert result.passed is False


def test_reload_diagnostic_wire_record_excludes_computed_passed_field() -> None:
    result = CheckpointReloadResult(
        missing=(),
        unexpected=(),
        state_digest="a" * 64,
        reference_state_digest="a" * 64,
        state_equal=True,
        input_hashes=DraftProbeInputHashes(
            noise_embedding="1" * 64,
            target_hidden="2" * 64,
            position_ids="3" * 64,
        ),
        inputs_equal=True,
        reference_execution_state=execution_state(),
        observed_execution_state=execution_state(),
        reference_repeat_equal=True,
        observed_repeat_equal=True,
        output_equal=True,
        output_allclose=True,
        output_mismatch_count=0,
        output_max_abs_difference=0,
        output_mean_abs_difference=0,
    )

    encoded = probe_wire_json(result)

    assert '"passed"' not in encoded
    assert CheckpointReloadResult.model_validate_json(encoded).passed is True


def test_probe_wire_record_excludes_nested_computed_fields() -> None:
    result = CapturedBatchOptimizerResult.model_validate(
        {
            "sample_id": "sample",
            "input_ids_shape": [1, 4],
            "loss_mask_shape": [1, 4],
            "hidden_states_shape": [1, 4, 10240],
            "loss": 1.0,
            "accuracy": 0.0,
            "accuracy_denom": 1,
            "gradient_parameters": [
                "draft_model.layers.0.self_attn.qkv_proj.weight",
                "draft_model.layers.0.self_attn.g_proj.weight",
                "draft_model.fc.weight",
                "draft_model.aux_hidden_norms.0.weight",
            ],
            "optimizer_state_entries": 58,
            "changed_parameter": "draft_model.layers.0.self_attn.g_proj.weight",
            "parameter_delta": 1.0,
            "checkpoint_hashes": {
                "config": "1" * 64,
                "weights": "2" * 64,
                "optimizer": "3" * 64,
                "random_state": "4" * 64,
                "manifest": "5" * 64,
            },
            "checkpoint_path": "/checkpoints/candidate",
            "pre_save_state_digest": "6" * 64,
            "training_cursor": 1,
            "reload": {
                "missing": [],
                "unexpected": [],
                "state_digest": "6" * 64,
                "reference_state_digest": "6" * 64,
                "state_equal": True,
                "input_hashes": {
                    "noise_embedding": "7" * 64,
                    "target_hidden": "8" * 64,
                    "position_ids": "9" * 64,
                },
                "inputs_equal": True,
                "reference_execution_state": execution_state().model_dump(mode="json"),
                "observed_execution_state": execution_state().model_dump(mode="json"),
                "reference_repeat_equal": True,
                "observed_repeat_equal": True,
                "output_equal": True,
                "output_allclose": True,
                "output_mismatch_count": 0,
                "output_max_abs_difference": 0,
                "output_mean_abs_difference": 0,
            },
            "released": True,
            "release_pending": 0,
        }
    )

    encoded = probe_wire_json(result)

    assert '"passed"' not in encoded
    assert CapturedBatchOptimizerResult.model_validate_json(encoded).passed is True
