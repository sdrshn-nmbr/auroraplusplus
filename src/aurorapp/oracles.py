from typing import Any

import numpy as np
from pydantic import Field

from aurorapp.models import StrictModel


class LosslessObservation(StrictModel):
    token_ids: tuple[int, ...]
    text: str
    stop_reason: str
    tool_structure: dict[str, Any]
    stream_chunks: tuple[str, ...]


class LosslessResult(StrictModel):
    passed: bool
    mismatches: tuple[str, ...]


def evaluate_lossless(
    baseline: LosslessObservation,
    challenger: LosslessObservation,
) -> LosslessResult:
    comparisons = {
        "token-ids": baseline.token_ids == challenger.token_ids,
        "text": baseline.text == challenger.text,
        "stop-reason": baseline.stop_reason == challenger.stop_reason,
        "tool-structure": baseline.tool_structure == challenger.tool_structure,
        "stream-chunks": baseline.stream_chunks == challenger.stream_chunks,
    }
    mismatches = tuple(name for name, matches in comparisons.items() if not matches)
    return LosslessResult(passed=not mismatches, mismatches=mismatches)


class PairedLatencyObservation(StrictModel):
    request_id: str
    baseline_output_tokens: int = Field(gt=0)
    challenger_output_tokens: int = Field(gt=0)
    baseline_seconds: float = Field(gt=0)
    challenger_seconds: float = Field(gt=0)
    baseline_p95_seconds: float = Field(gt=0)
    challenger_p95_seconds: float = Field(gt=0)
    baseline_p99_seconds: float = Field(gt=0)
    challenger_p99_seconds: float = Field(gt=0)


class PairedPerformanceStatistics(StrictModel):
    output_rate_lcb_improvement: float
    p95_latency_upper_regression: float
    p99_latency_upper_regression: float
    request_count: int
    resamples: int
    seed: int


def paired_performance_statistics(
    observations: tuple[PairedLatencyObservation, ...],
    resamples: int,
    seed: int,
) -> PairedPerformanceStatistics:
    if not observations:
        raise ValueError("paired observations cannot be empty")
    request_ids = [observation.request_id for observation in observations]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("paired request IDs must be unique")
    if resamples < 100:
        raise ValueError("at least 100 resamples are required")
    baseline_rates = np.array(
        [
            observation.baseline_output_tokens / observation.baseline_seconds
            for observation in observations
        ]
    )
    challenger_rates = np.array(
        [
            observation.challenger_output_tokens / observation.challenger_seconds
            for observation in observations
        ]
    )
    p95_regressions = np.array(
        [
            observation.challenger_p95_seconds / observation.baseline_p95_seconds - 1
            for observation in observations
        ]
    )
    p99_regressions = np.array(
        [
            observation.challenger_p99_seconds / observation.baseline_p99_seconds - 1
            for observation in observations
        ]
    )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(observations), size=(resamples, len(observations)))
    rate_improvements = (
        challenger_rates[sample_indices].mean(axis=1) / baseline_rates[sample_indices].mean(axis=1)
        - 1
    )
    p95_samples = p95_regressions[sample_indices].mean(axis=1)
    p99_samples = p99_regressions[sample_indices].mean(axis=1)
    return PairedPerformanceStatistics(
        output_rate_lcb_improvement=float(np.quantile(rate_improvements, 0.05)),
        p95_latency_upper_regression=float(np.quantile(p95_samples, 0.95)),
        p99_latency_upper_regression=float(np.quantile(p99_samples, 0.95)),
        request_count=len(observations),
        resamples=resamples,
        seed=seed,
    )
