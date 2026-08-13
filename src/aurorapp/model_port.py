import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, computed_field, model_validator

from aurorapp.models import GitRevision, Sha256, StrictModel


class GenerationFinishReason(StrictModel):
    type: str = Field(min_length=1)
    length: int | None = Field(default=None, ge=0)


class SpeculativeTelemetry(StrictModel):
    proposed_drafts: int = Field(ge=0)
    accepted_drafts: int = Field(ge=0)
    verify_count: int = Field(ge=0)
    accept_histogram: tuple[int, ...]

    @model_validator(mode="after")
    def counts_are_consistent(self) -> "SpeculativeTelemetry":
        if self.accepted_drafts > self.proposed_drafts:
            raise ValueError("accepted drafts cannot exceed proposed drafts")
        if any(count < 0 for count in self.accept_histogram):
            raise ValueError("accept histogram counts cannot be negative")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return self.proposed_drafts > 0 and self.verify_count > 0


class CandidateSpeculativeServingResult(StrictModel):
    target_repository: str = Field(min_length=1)
    target_revision: GitRevision
    parent_draft_repository: str = Field(min_length=1)
    parent_draft_revision: GitRevision
    candidate_checkpoint_path: str = Field(min_length=1)
    candidate_manifest_hash: Sha256
    candidate_weights_hash: Sha256
    request_hash: Sha256
    target_output_ids: tuple[int, ...] = Field(min_length=1)
    candidate_output_ids: tuple[int, ...] = Field(min_length=1)
    target_text: str
    candidate_text: str
    target_finish_reason: GenerationFinishReason
    candidate_finish_reason: GenerationFinishReason
    speculative_telemetry: SpeculativeTelemetry
    target_server_healthy: bool
    candidate_server_healthy: bool
    draft_checkpoint_loaded: bool
    target_cleanup_passed: bool
    candidate_cleanup_passed: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return (
            self.target_server_healthy
            and self.candidate_server_healthy
            and self.draft_checkpoint_loaded
            and self.target_cleanup_passed
            and self.candidate_cleanup_passed
            and self.target_output_ids == self.candidate_output_ids
            and self.target_text == self.candidate_text
            and self.target_finish_reason == self.candidate_finish_reason
            and self.speculative_telemetry.passed
        )


class SampledOutput(StrictModel):
    output_ids: tuple[int, ...] = Field(min_length=1)
    text: str
    finish_reason: GenerationFinishReason


class SampledSeedRepeatability(StrictModel):
    sampling_seed: int = Field(ge=0)
    repetitions: tuple[SampledOutput, SampledOutput]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def repeatable(self) -> bool:
        return self.repetitions[0] == self.repetitions[1]


class TargetSamplingDeterminismResult(StrictModel):
    target_repository: str = Field(min_length=1)
    target_revision: GitRevision
    request_contract_hash: Sha256
    moe_runner_backend: Literal["auto", "triton"]
    deterministic_inference_enabled: bool
    radix_cache_disabled: bool
    cases: tuple[SampledSeedRepeatability, ...] = Field(min_length=2)
    server_healthy: bool
    cleanup_passed: bool

    @model_validator(mode="after")
    def seeds_are_unique(self) -> "TargetSamplingDeterminismResult":
        seeds = [case.sampling_seed for case in self.cases]
        if len(set(seeds)) != len(seeds):
            raise ValueError("target determinism cases must use unique seeds")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def outputs_vary_across_seeds(self) -> bool:
        outputs = {
            (case.repetitions[0].output_ids, case.repetitions[0].text)
            for case in self.cases
        }
        return len(outputs) > 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return (
            self.deterministic_inference_enabled
            and self.radix_cache_disabled
            and self.server_healthy
            and self.cleanup_passed
            and self.outputs_vary_across_seeds
            and all(case.repeatable for case in self.cases)
        )


class SampledSeedParity(StrictModel):
    sampling_seed: int = Field(ge=0)
    target: tuple[SampledOutput, SampledOutput]
    candidate: tuple[SampledOutput, SampledOutput]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def target_repeatable(self) -> bool:
        return self.target[0] == self.target[1]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def candidate_repeatable(self) -> bool:
        return self.candidate[0] == self.candidate[1]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def target_candidate_equal(self) -> bool:
        return self.target[0] == self.candidate[0]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return self.target_repeatable and self.candidate_repeatable and self.target_candidate_equal


class SampledRngParityResult(StrictModel):
    target_repository: str = Field(min_length=1)
    target_revision: GitRevision
    parent_draft_repository: str = Field(min_length=1)
    parent_draft_revision: GitRevision
    candidate_checkpoint_path: str = Field(min_length=1)
    candidate_manifest_hash: Sha256
    candidate_weights_hash: Sha256
    request_contract_hash: Sha256
    deterministic_inference_enabled: bool
    cases: tuple[SampledSeedParity, ...] = Field(min_length=2)
    speculative_telemetry: SpeculativeTelemetry
    target_server_healthy: bool
    candidate_server_healthy: bool
    target_cleanup_passed: bool
    candidate_cleanup_passed: bool

    @model_validator(mode="after")
    def seeds_are_unique(self) -> "SampledRngParityResult":
        seeds = [case.sampling_seed for case in self.cases]
        if len(set(seeds)) != len(seeds):
            raise ValueError("sampled parity cases must use unique seeds")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def outputs_vary_across_seeds(self) -> bool:
        outputs = {(case.target[0].output_ids, case.target[0].text) for case in self.cases}
        return len(outputs) > 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return (
            self.deterministic_inference_enabled
            and self.target_server_healthy
            and self.candidate_server_healthy
            and self.target_cleanup_passed
            and self.candidate_cleanup_passed
            and self.speculative_telemetry.passed
            and self.outputs_vary_across_seeds
            and all(case.passed for case in self.cases)
        )


class TokenizerTemplateIdentityResult(StrictModel):
    target_repository: str = Field(min_length=1)
    target_revision: GitRevision
    parent_draft_repository: str = Field(min_length=1)
    parent_draft_revision: GitRevision
    target_tokenizer_file_hashes: dict[str, Sha256] = Field(min_length=1)
    expected_target_tokenizer_file_hashes: dict[str, Sha256] = Field(min_length=1)
    draft_repository_files: tuple[str, ...] = Field(min_length=1)
    draft_tokenizer_overrides: tuple[str, ...]
    loaded_chat_template_hash: Sha256
    vocabulary_hash: Sha256
    rendered_prompt_hash: Sha256
    rendered_token_ids_hash: Sha256
    vocabulary_size: int = Field(gt=0)
    tokenizer_length: int = Field(gt=0)
    special_token_ids: dict[str, int] = Field(min_length=1)
    runtime_tokenizer_path: str = Field(min_length=1)
    draft_worker_skips_tokenizer: bool
    sglang_tokenizer_source_hash: Sha256

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        template_hash = self.target_tokenizer_file_hashes.get("chat_template.jinja")
        return (
            self.target_tokenizer_file_hashes == self.expected_target_tokenizer_file_hashes
            and not self.draft_tokenizer_overrides
            and template_hash == self.loaded_chat_template_hash
            and self.runtime_tokenizer_path == self.target_repository
            and self.draft_worker_skips_tokenizer
            and self.tokenizer_length >= self.vocabulary_size
        )


class ParentDrafterRestoreResult(StrictModel):
    candidate_serving_evidence_hash: Sha256
    candidate_manifest_hash: Sha256
    parent_draft_repository: str = Field(min_length=1)
    parent_draft_revision: GitRevision
    request_hash: Sha256
    expected_output_ids: tuple[int, ...] = Field(min_length=1)
    parent_output_ids: tuple[int, ...] = Field(min_length=1)
    expected_text: str
    parent_text: str
    expected_finish_reason: GenerationFinishReason
    parent_finish_reason: GenerationFinishReason
    speculative_telemetry: SpeculativeTelemetry
    server_healthy: bool
    cleanup_passed: bool

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return (
            self.server_healthy
            and self.cleanup_passed
            and self.expected_output_ids == self.parent_output_ids
            and self.expected_text == self.parent_text
            and self.expected_finish_reason == self.parent_finish_reason
            and self.speculative_telemetry.passed
        )


def candidate_serving_from_probe_payload(
    payload: Mapping[str, object],
) -> CandidateSpeculativeServingResult:
    clean = dict(payload)
    clean.pop("passed", None)
    telemetry = clean.get("speculative_telemetry")
    if not isinstance(telemetry, Mapping):
        raise ValueError("candidate serving payload has no speculative telemetry")
    clean["speculative_telemetry"] = {
        key: value for key, value in telemetry.items() if key != "passed"
    }
    return CandidateSpeculativeServingResult.model_validate(clean)


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


class PhysicalModelPortResult(StrictModel):
    architecture: Literal["DFlashLagunaForCausalLM"]
    checkpoint: CheckpointHeaderResult
    loading_missing: tuple[str, ...]
    loading_unexpected: tuple[str, ...]
    loading_mismatched: tuple[str, ...]
    forward_shape: tuple[int, int, int]
    loss_finite: bool
    gradient_parameters: tuple[str, ...]
    optimizer_state_entries: int = Field(ge=0)
    changed_parameter: Literal["layers.0.self_attn.g_proj.weight"]
    parameter_delta: float = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        required_gradients = {
            "layers.0.self_attn.qkv_proj.weight",
            "layers.0.self_attn.g_proj.weight",
            "fc.weight",
            "aux_hidden_norms.0.weight",
        }
        return (
            self.checkpoint.passed
            and not self.loading_missing
            and not self.loading_unexpected
            and not self.loading_mismatched
            and self.forward_shape == (1, 4, 2048)
            and self.loss_finite
            and required_gradients.issubset(self.gradient_parameters)
            and self.optimizer_state_entries > 0
            and math.isfinite(self.parameter_delta)
            and self.parameter_delta > 0
        )


class CapturedBatchOptimizerResult(StrictModel):
    sample_id: str = Field(min_length=1)
    input_ids_shape: tuple[int, int]
    loss_mask_shape: tuple[int, int]
    hidden_states_shape: tuple[int, int, int]
    loss: float
    accuracy: float
    accuracy_denom: int = Field(ge=0)
    gradient_parameters: tuple[str, ...]
    optimizer_state_entries: int = Field(ge=0)
    changed_parameter: Literal["draft_model.layers.0.self_attn.g_proj.weight"]
    parameter_delta: float = Field(ge=0)
    checkpoint_hashes: dict[str, Sha256]
    source_config_hash: Sha256
    checkpoint_path: str = Field(min_length=1)
    pre_save_state_digest: Sha256
    training_cursor: Literal[1]
    reload: "CheckpointReloadResult"
    released: bool
    release_pending: int = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        required_gradients = {
            "draft_model.layers.0.self_attn.qkv_proj.weight",
            "draft_model.layers.0.self_attn.g_proj.weight",
            "draft_model.fc.weight",
            "draft_model.aux_hidden_norms.0.weight",
        }
        required_checkpoint_files = {
            "config",
            "weights",
            "optimizer",
            "random_state",
            "manifest",
        }
        return (
            bool(self.sample_id)
            and self.input_ids_shape == (1, 4)
            and self.loss_mask_shape == (1, 4)
            and self.hidden_states_shape == (1, 4, 10240)
            and math.isfinite(self.loss)
            and self.loss > 0
            and math.isfinite(self.accuracy)
            and 0 <= self.accuracy <= 1
            and self.accuracy_denom > 0
            and required_gradients.issubset(self.gradient_parameters)
            and self.optimizer_state_entries > 0
            and math.isfinite(self.parameter_delta)
            and self.parameter_delta > 0
            and required_checkpoint_files == set(self.checkpoint_hashes)
            and self.checkpoint_hashes["config"] == self.source_config_hash
            and self.reload.reference_state_digest == self.pre_save_state_digest
            and self.reload.passed
            and self.released
            and self.release_pending == 0
        )


class CheckpointReloadResult(StrictModel):
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    state_digest: Sha256
    reference_state_digest: Sha256
    state_equal: bool
    input_hashes: "DraftProbeInputHashes"
    inputs_equal: bool
    reference_execution_state: "DraftExecutionState"
    observed_execution_state: "DraftExecutionState"
    reference_repeat_equal: bool
    observed_repeat_equal: bool
    output_equal: bool
    output_allclose: bool
    output_mismatch_count: int = Field(ge=0)
    output_max_abs_difference: float = Field(ge=0)
    output_mean_abs_difference: float = Field(ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return (
            not self.missing
            and not self.unexpected
            and self.state_equal
            and self.state_digest == self.reference_state_digest
            and self.inputs_equal
            and self.reference_execution_state == self.observed_execution_state
            and self.reference_repeat_equal
            and self.observed_repeat_equal
            and self.output_equal
            and self.output_allclose
            and self.output_mismatch_count == 0
            and self.output_max_abs_difference == 0
            and self.output_mean_abs_difference == 0
        )


class CheckpointReferenceResult(StrictModel):
    state_digest: Sha256
    input_hashes: "DraftProbeInputHashes"
    execution_state: "DraftExecutionState"
    repeat_equal: bool
    reference_path: str = Field(min_length=1)


class DraftProbeInputHashes(StrictModel):
    noise_embedding: Sha256
    target_hidden: Sha256
    position_ids: Sha256


class DraftExecutionState(StrictModel):
    nonpersistent_buffer_digest: Sha256
    parameter_layout_digest: Sha256
    module_modes_digest: Sha256
    runtime_config_digest: Sha256
    float32_matmul_precision: str = Field(min_length=1)
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    deterministic_algorithms: bool
    cublas_allow_tf32: bool
    cudnn_allow_tf32: bool


class DraftRuntimeConfig(StrictModel):
    model_type: Literal["laguna"]
    hidden_size: int = Field(gt=0)
    intermediate_size: int = Field(gt=0)
    num_hidden_layers: int = Field(gt=0)
    num_attention_heads: int = Field(gt=0)
    num_key_value_heads: int = Field(gt=0)
    head_dim: int = Field(gt=0)
    rms_norm_eps: float = Field(gt=0)
    max_position_embeddings: int = Field(gt=0)
    sliding_window: int = Field(gt=0)
    rope_theta: float = Field(gt=0)
    attention_dropout: float = Field(ge=0)
    attention_implementation: str = Field(min_length=1)
    gating: Literal["per-head"]
    layer_types: tuple[Literal["sliding_attention"], ...]
    dflash_config: LagunaDFlashMethodContract


def draft_runtime_config(
    config: Mapping[str, object],
    *,
    attention_implementation: str,
) -> DraftRuntimeConfig:
    fields = DraftRuntimeConfig.model_fields
    payload = {name: config[name] for name in fields if name in config}
    payload["attention_implementation"] = attention_implementation
    return DraftRuntimeConfig.model_validate(payload)


def probe_wire_json(result: StrictModel) -> str:
    return result.model_dump_json(exclude_computed_fields=True)


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
    header: Mapping[str, object],
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
