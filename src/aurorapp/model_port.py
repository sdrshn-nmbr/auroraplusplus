import json
import struct
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from aurorapp.models import StrictModel


class LagunaDFlashMethodContract(StrictModel):
    block_size: int = Field(ge=2)
    mask_token_id: int = Field(ge=0)
    num_target_layers: int = Field(gt=0)
    target_layer_ids: tuple[int, ...]
    causal: Literal[True]


class LagunaDFlashConfigContract(StrictModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    architectures: tuple[Literal["DFlashLagunaForCausalLM"], ...]
    model_type: Literal["laguna"]
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    attention_bias: Literal[False]
    gating: Literal["per-head"]
    layer_types: tuple[Literal["sliding_attention"], ...]
    dflash_config: LagunaDFlashMethodContract

    @model_validator(mode="after")
    def dimensions_are_consistent(self) -> "LagunaDFlashConfigContract":
        if self.architectures != ("DFlashLagunaForCausalLM",):
            raise ValueError("official port requires one Laguna DFlash architecture")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one item per draft layer")
        if len(self.dflash_config.target_layer_ids) != self.num_hidden_layers:
            raise ValueError("target_layer_ids must contain one item per draft layer")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("attention heads must be divisible by key-value heads")
        return self


class CheckpointTensor(StrictModel):
    dtype: Literal["BF16"]
    shape: tuple[int, ...]


class CheckpointHeaderResult(StrictModel):
    passed: bool
    expected_count: int
    observed_count: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_or_dtype_mismatches: tuple[str, ...]


def checkpoint_contract(
    config: LagunaDFlashConfigContract,
) -> dict[str, CheckpointTensor]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    query_width = config.num_attention_heads * config.head_dim
    key_value_width = config.num_key_value_heads * config.head_dim
    tensors: dict[str, CheckpointTensor] = {}
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        shapes = {
            f"{prefix}.input_layernorm.weight": (hidden,),
            f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
            f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
            f"{prefix}.post_attention_layernorm.weight": (hidden,),
            f"{prefix}.self_attn.g_proj.weight": (
                config.num_attention_heads,
                hidden,
            ),
            f"{prefix}.self_attn.k_norm.weight": (config.head_dim,),
            f"{prefix}.self_attn.o_proj.weight": (hidden, query_width),
            f"{prefix}.self_attn.q_norm.weight": (config.head_dim,),
            f"{prefix}.self_attn.qkv_proj.weight": (
                query_width + 2 * key_value_width,
                hidden,
            ),
        }
        tensors.update(
            {name: CheckpointTensor(dtype="BF16", shape=shape) for name, shape in shapes.items()}
        )
    for index in range(len(config.dflash_config.target_layer_ids)):
        tensors[f"aux_hidden_norms.{index}.weight"] = CheckpointTensor(
            dtype="BF16", shape=(hidden,)
        )
    tensors.update(
        {
            "fc.weight": CheckpointTensor(
                dtype="BF16",
                shape=(hidden, hidden * len(config.dflash_config.target_layer_ids)),
            ),
            "hidden_norm.weight": CheckpointTensor(dtype="BF16", shape=(hidden,)),
            "norm.weight": CheckpointTensor(dtype="BF16", shape=(hidden,)),
        }
    )
    return tensors


def validate_checkpoint_header(
    config: LagunaDFlashConfigContract,
    header: dict[str, object],
) -> CheckpointHeaderResult:
    expected = checkpoint_contract(config)
    observed_names = set(header) - {"__metadata__"}
    expected_names = set(expected)
    mismatches: list[str] = []
    for name in sorted(expected_names & observed_names):
        raw = header[name]
        if not isinstance(raw, dict):
            mismatches.append(name)
            continue
        if (
            raw.get("dtype") != expected[name].dtype
            or tuple(raw.get("shape", ())) != expected[name].shape
        ):
            mismatches.append(name)
    missing = tuple(sorted(expected_names - observed_names))
    unexpected = tuple(sorted(observed_names - expected_names))
    shape_or_dtype_mismatches = tuple(mismatches)
    return CheckpointHeaderResult(
        passed=not missing and not unexpected and not shape_or_dtype_mismatches,
        expected_count=len(expected_names),
        observed_count=len(observed_names),
        missing=missing,
        unexpected=unexpected,
        shape_or_dtype_mismatches=shape_or_dtype_mismatches,
    )


def read_safetensors_header(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            raise ValueError("safetensors file has no complete header length")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size <= 0 or header_size > 16 * 1024 * 1024:
            raise ValueError("safetensors header length is outside the safe bound")
        raw_header = stream.read(header_size)
    if len(raw_header) != header_size:
        raise ValueError("safetensors file has a partial header")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    return header
