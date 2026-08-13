import json

from aurorapp.model_port import (
    CapturedBatchOptimizerResult,
    LagunaDFlashConfigContract,
    PhysicalModelPortResult,
    checkpoint_contract,
    validate_checkpoint_header,
)


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
        "training_cursor": 1,
        "reload_missing": [],
        "reload_unexpected": [],
        "reload_output_equal": True,
        "released": True,
        "release_pending": 0,
    }

    result = CapturedBatchOptimizerResult.model_validate(payload)

    assert result.passed is True
    payload["checkpoint_hashes"].pop("random_state")
    assert CapturedBatchOptimizerResult.model_validate(payload).passed is False
