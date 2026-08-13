import numpy as np

from aurorapp.oracles import (
    LosslessObservation,
    PairedLatencyObservation,
    evaluate_lossless,
    paired_performance_statistics,
)


def test_lossless_oracle_compares_every_declared_surface() -> None:
    baseline = LosslessObservation(
        token_ids=(1, 2), text="ok", stop_reason="stop", tool_structure={}, stream_chunks=("o", "k")
    )
    changed = baseline.model_copy(update={"stream_chunks": ("ok",)})

    result = evaluate_lossless(baseline, changed)

    assert result.passed is False
    assert result.mismatches == ("stream-chunks",)


def test_paired_bootstrap_is_reproducible() -> None:
    observations = tuple(
        PairedLatencyObservation(
            request_id=f"request-{index}",
            baseline_output_tokens=100,
            challenger_output_tokens=100,
            baseline_seconds=2.0,
            challenger_seconds=1.8,
            baseline_p95_seconds=2.1,
            challenger_p95_seconds=2.0,
            baseline_p99_seconds=2.2,
            challenger_p99_seconds=2.1,
        )
        for index in range(32)
    )

    first = paired_performance_statistics(observations, resamples=1000, seed=17)
    second = paired_performance_statistics(observations, resamples=1000, seed=17)

    assert first == second
    assert first.output_rate_lcb_improvement > 0


def test_performance_oracle_rejects_unpaired_request_ids() -> None:
    observations = (
        PairedLatencyObservation(
            request_id="duplicate",
            baseline_output_tokens=1,
            challenger_output_tokens=1,
            baseline_seconds=1,
            challenger_seconds=1,
            baseline_p95_seconds=1,
            challenger_p95_seconds=1,
            baseline_p99_seconds=1,
            challenger_p99_seconds=1,
        ),
    ) * 2

    with np.testing.assert_raises_regex(ValueError, "unique"):
        paired_performance_statistics(observations, resamples=100, seed=1)
