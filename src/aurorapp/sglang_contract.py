from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from aurorapp.canonical import canonical_sha256
from aurorapp.models import GitRevision, Sha256, StrictModel
from aurorapp.sampled_distribution import (
    DistributionStratumSpec,
    SampledDistributionRule,
    SampledDistributionSuiteRule,
)

SGLANG_SERVER_RANDOM_SEED = 20260812
SAMPLED_GENERATION_SEEDS = (17, 42, 20260812)
LAGUNA_TOKENIZER_FILE_HASHES = {
    "chat_template.jinja": "2d3c724b3c2e9eb71fe9ccc5423ff268a370a8bfa89e9238b6de14fe000825c8",
    "special_tokens_map.json": "70cd3459fde61761e9440751a590e89a108c09b1803cc7727f5ad1ed1ea6122b",
    "tokenizer.json": "807c53a95141e77c14e45f68c51db3f84d2ea6b555a6ea832bc99c88dae6a279",
    "tokenizer_config.json": "7458ce192bcd810dd43cad15c1996711168f2cc12d5400dac247da5033d6c283",
}


def sampled_distribution_suite_rule() -> SampledDistributionSuiteRule:
    return SampledDistributionSuiteRule(
        distribution=SampledDistributionRule(
            samples_per_arm=64,
            bootstrap_resamples=2_000,
            equivalence_margin=0.03,
            bootstrap_seed=20260813,
        ),
        strata=(
            DistributionStratumSpec(
                stratum_id="code-temperature-0.8-top-p-0.95",
                prompt="Write a Python function that adds two tensors.",
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=32,
            ),
            DistributionStratumSpec(
                stratum_id="reasoning-temperature-1.0-top-p-1.0",
                prompt=(
                    "Explain how to verify that a tiled matrix multiplication kernel "
                    "is correct."
                ),
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=32,
            ),
        ),
        power_trials=40,
        corruption_fraction=0.25,
        minimum_power=0.9,
        calibration_seed=41,
    )


class SamplingDecisionDomain(StrEnum):
    TARGET_TOKEN = "target-token"
    DFLASH_TARGET_TOKEN = "dflash-target-token"


class PositionSamplingContract(StrictModel):
    version: Literal["position-v1"] = "position-v1"
    sampling_seed: int = Field(ge=0)
    absolute_position: int = Field(ge=0)
    domain: SamplingDecisionDomain

    @property
    def coupled_domain(self) -> str:
        if self.domain in {
            SamplingDecisionDomain.TARGET_TOKEN,
            SamplingDecisionDomain.DFLASH_TARGET_TOKEN,
        }:
            return "target-token"
        raise ValueError(f"unsupported sampling decision domain: {self.domain}")

    @property
    def decision_key(self) -> Sha256:
        return canonical_sha256(
            {
                "version": self.version,
                "sampling_seed": self.sampling_seed,
                "absolute_position": self.absolute_position,
                "domain": self.coupled_domain,
            }
        )


class MooncakeReplayEnvelope(StrictModel):
    version: Literal["mooncake-replay-v1"] = "mooncake-replay-v1"
    run_id: Sha256
    request_id: str = Field(min_length=1)
    target_repository: str = Field(min_length=1)
    target_revision: GitRevision
    tokenizer_hash: Sha256
    sampling_contract_hash: Sha256
    input_ids_hash: Sha256
    loss_mask_hash: Sha256
    hidden_states_hash: Sha256
    hidden_state_shape: tuple[int, int, int]
    hidden_state_dtype: Literal["bfloat16"]
    aux_layer_ids: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_laguna_capture_shape(self) -> "MooncakeReplayEnvelope":
        if self.aux_layer_ids != (1, 13, 25, 33, 39):
            raise ValueError("Mooncake replay must contain the official Laguna DFlash layers")
        if self.hidden_state_shape[0] != 1 or self.hidden_state_shape[2] != 10240:
            raise ValueError("Mooncake replay has the wrong Laguna hidden-state shape")
        return self


def greedy_generation_request(text: str, max_new_tokens: int) -> dict[str, Any]:
    if not text:
        raise ValueError("generation text cannot be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    return {
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
    }


def sampled_generation_request(
    text: str,
    max_new_tokens: int,
    *,
    sampling_seed: int,
) -> dict[str, Any]:
    if not text:
        raise ValueError("generation text cannot be empty")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if sampling_seed < 0:
        raise ValueError("sampling_seed cannot be negative")
    return {
        "text": text,
        "sampling_params": {
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": max_new_tokens,
            "sampling_seed": sampling_seed,
        },
    }


def distribution_generation_request(
    stratum: DistributionStratumSpec,
    *,
    sampling_seed: int,
) -> dict[str, Any]:
    if sampling_seed < 0:
        raise ValueError("sampling_seed cannot be negative")
    return {
        "text": stratum.prompt,
        "sampling_params": {
            "temperature": stratum.temperature,
            "top_p": stratum.top_p,
            "max_new_tokens": stratum.max_new_tokens,
            "sampling_seed": sampling_seed,
        },
    }
