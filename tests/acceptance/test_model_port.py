import json

from aurorapp.model_port import (
    LagunaDFlashConfigContract,
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
