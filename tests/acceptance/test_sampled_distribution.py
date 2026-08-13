from aurorapp.sampled_distribution import (
    DistributionArm,
    DistributionSample,
    SampledDistributionRule,
    calibrate_distribution_power,
    evaluate_sampled_distribution,
)


def sample(token: int, suffix: int) -> DistributionSample:
    return DistributionSample(
        sampling_seed=(token * 10_000) + suffix,
        output_ids=(token, suffix),
        text=f"{token}-{suffix}",
        finish_reason="length",
    )


def arm(name: str, token: int, count: int = 64) -> DistributionArm:
    return DistributionArm(
        arm_id=name,
        server_start_id=f"start-{name}",
        samples=tuple(sample(token, index % 4) for index in range(count)),
    )


def test_distributional_gate_accepts_candidate_inside_target_self_variation() -> None:
    rule = SampledDistributionRule(
        samples_per_arm=64,
        bootstrap_resamples=400,
        equivalence_margin=0.03,
        bootstrap_seed=20260813,
    )
    result = evaluate_sampled_distribution(
        rule,
        target_a=arm("target-a", 1),
        target_b=arm("target-b", 1),
        candidate_a=arm("candidate-a", 1),
        candidate_b=arm("candidate-b", 1),
    )

    assert result.passed
    assert result.upper_confidence_bound <= rule.equivalence_margin


def test_distributional_gate_rejects_candidate_shift() -> None:
    rule = SampledDistributionRule(
        samples_per_arm=64,
        bootstrap_resamples=400,
        equivalence_margin=0.03,
        bootstrap_seed=20260813,
    )
    result = evaluate_sampled_distribution(
        rule,
        target_a=arm("target-a", 1),
        target_b=arm("target-b", 1),
        candidate_a=arm("candidate-a", 99),
        candidate_b=arm("candidate-b", 99),
    )

    assert not result.passed
    assert result.upper_confidence_bound > rule.equivalence_margin


def test_distributional_gate_requires_distinct_fresh_server_starts() -> None:
    rule = SampledDistributionRule(
        samples_per_arm=64,
        bootstrap_resamples=400,
        equivalence_margin=0.03,
        bootstrap_seed=20260813,
    )
    duplicate = arm("target-a", 1)
    target_b = arm("target-b", 1).model_copy(
        update={"server_start_id": duplicate.server_start_id}
    )

    try:
        evaluate_sampled_distribution(
            rule,
            target_a=duplicate,
            target_b=target_b,
            candidate_a=arm("candidate-a", 1),
            candidate_b=arm("candidate-b", 1),
        )
    except ValueError as error:
        assert "fresh server starts" in str(error)
    else:
        raise AssertionError("duplicate server starts must fail")


def test_signed_distribution_rule_has_power_against_declared_corruptions() -> None:
    rule = SampledDistributionRule(
        samples_per_arm=64,
        bootstrap_resamples=400,
        equivalence_margin=0.03,
        bootstrap_seed=20260813,
    )

    calibration = calibrate_distribution_power(
        rule,
        trials=20,
        corruption_fraction=0.25,
        minimum_power=0.9,
        calibration_seed=41,
    )

    assert calibration.passed
    assert {case.corruption for case in calibration.cases} == {
        "first-token-substitution",
        "early-truncation",
    }
