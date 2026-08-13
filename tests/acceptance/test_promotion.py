import math

import pytest
from pydantic import ValidationError

from aurorapp.models import ConcurrencyResult, PairedEvaluation, PromotionLimits
from aurorapp.promotion import decide_promotion


def good_result(concurrency: int) -> ConcurrencyResult:
    return ConcurrencyResult(
        concurrency=concurrency,
        matched_requests=256,
        output_rate_lcb_improvement=0.031,
        p95_latency_upper_regression=0.009,
        p99_latency_upper_regression=0.019,
        parity_mismatches=0,
        correctness_regressions=0,
        cold_load_successes=3,
        health_failures=[],
    )


def test_conservative_gate_accepts_only_both_concurrencies() -> None:
    evaluation = PairedEvaluation(results=[good_result(1), good_result(4)])

    decision = decide_promotion(evaluation, PromotionLimits())

    assert decision.eligible is True
    assert decision.reasons == []


def test_one_output_mismatch_blocks_promotion() -> None:
    broken = good_result(4).model_copy(update={"parity_mismatches": 1})

    decision = decide_promotion(
        PairedEvaluation(results=[good_result(1), broken]),
        PromotionLimits(),
    )

    assert decision.eligible is False
    assert any("parity" in reason for reason in decision.reasons)


def test_tail_regression_blocks_promotion() -> None:
    broken = good_result(1).model_copy(update={"p99_latency_upper_regression": 0.021})

    decision = decide_promotion(
        PairedEvaluation(results=[broken, good_result(4)]),
        PromotionLimits(),
    )

    assert decision.eligible is False
    assert any("p99" in reason for reason in decision.reasons)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_measurement_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        ConcurrencyResult(
            concurrency=1,
            matched_requests=256,
            output_rate_lcb_improvement=value,
            p95_latency_upper_regression=0.0,
            p99_latency_upper_regression=0.0,
            parity_mismatches=0,
            correctness_regressions=0,
            cold_load_successes=3,
            health_failures=[],
        )


def test_duplicate_concurrency_is_rejected_by_gate() -> None:
    decision = decide_promotion(
        PairedEvaluation(results=[good_result(1), good_result(1), good_result(4)]),
        PromotionLimits(),
    )

    assert decision.eligible is False
    assert decision.reasons == ["duplicate concurrencies: [1]"]


def test_malformed_required_concurrencies_are_rejected() -> None:
    with pytest.raises(ValidationError):
        PromotionLimits(required_concurrencies=(1, 1))
