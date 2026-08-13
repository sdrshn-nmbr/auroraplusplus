import random
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, computed_field, model_validator

from aurorapp.models import StrictModel


class DistributionSample(StrictModel):
    sampling_seed: int = Field(ge=0)
    output_ids: tuple[int, ...] = Field(min_length=1)
    text: str
    finish_reason: str = Field(min_length=1)


class DistributionArm(StrictModel):
    arm_id: str = Field(min_length=1)
    server_start_id: str = Field(min_length=1)
    samples: tuple[DistributionSample, ...] = Field(min_length=1)


class SampledDistributionRule(StrictModel):
    version: Literal["sampled-distribution-v1"] = "sampled-distribution-v1"
    samples_per_arm: int = Field(ge=24)
    bootstrap_resamples: int = Field(ge=400)
    confidence_level: float = Field(default=0.95, ge=0.95, le=0.95)
    equivalence_margin: float = Field(gt=0, le=0.25)
    bootstrap_seed: int = Field(ge=0)
    distance: Literal["normalized-token-edit-energy"] = "normalized-token-edit-energy"


class SampledDistributionResult(StrictModel):
    rule: SampledDistributionRule
    target_self_discrepancy: float = Field(ge=0)
    target_candidate_discrepancy: float = Field(ge=0)
    observed_excess: float
    upper_confidence_bound: float
    server_start_ids: tuple[str, str, str, str]
    arm_sample_counts: tuple[int, int, int, int]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return self.upper_confidence_bound <= self.rule.equivalence_margin


class PowerCalibrationCase(StrictModel):
    corruption: Literal["first-token-substitution", "early-truncation"]
    corruption_fraction: float = Field(gt=0, le=1)
    trials: int = Field(ge=20)
    detections: int = Field(ge=0)

    @model_validator(mode="after")
    def detections_do_not_exceed_trials(self) -> "PowerCalibrationCase":
        if self.detections > self.trials:
            raise ValueError("power detections cannot exceed trials")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def power(self) -> float:
        return self.detections / self.trials


class DistributionPowerCalibration(StrictModel):
    minimum_power: float = Field(ge=0.8, le=1)
    cases: tuple[PowerCalibrationCase, ...] = Field(min_length=2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return all(case.power >= self.minimum_power for case in self.cases)


class DistributionStratumSpec(StrictModel):
    stratum_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    temperature: float = Field(gt=0, le=2)
    top_p: float = Field(gt=0, le=1)
    max_new_tokens: int = Field(ge=8, le=128)


class SampledDistributionSuiteRule(StrictModel):
    version: Literal["sampled-distribution-suite-v1"] = "sampled-distribution-suite-v1"
    distribution: SampledDistributionRule
    strata: tuple[DistributionStratumSpec, ...] = Field(min_length=2)
    power_trials: int = Field(ge=20)
    corruption_fraction: float = Field(gt=0, le=1)
    minimum_power: float = Field(ge=0.8, le=1)
    calibration_seed: int = Field(ge=0)

    @model_validator(mode="after")
    def strata_are_unique(self) -> "SampledDistributionSuiteRule":
        identifiers = [stratum.stratum_id for stratum in self.strata]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("distribution strata must be unique")
        return self


class DistributionStratumResult(StrictModel):
    stratum: DistributionStratumSpec
    result: SampledDistributionResult


class SampledDistributionSuiteResult(StrictModel):
    rule: SampledDistributionSuiteRule
    power_calibration: DistributionPowerCalibration
    strata: tuple[DistributionStratumResult, ...]
    target_cleanup_passed: bool
    candidate_cleanup_passed: bool
    candidate_proposed_drafts: int = Field(ge=0)
    candidate_verify_count: int = Field(ge=0)

    @model_validator(mode="after")
    def results_match_rule(self) -> "SampledDistributionSuiteResult":
        expected = tuple(value.stratum_id for value in self.rule.strata)
        observed = tuple(value.stratum.stratum_id for value in self.strata)
        if observed != expected:
            raise ValueError("sampled distribution results must match the signed strata")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return (
            self.power_calibration.passed
            and self.target_cleanup_passed
            and self.candidate_cleanup_passed
            and self.candidate_proposed_drafts > 0
            and self.candidate_verify_count > 0
            and all(value.result.passed for value in self.strata)
        )


def calibrate_distribution_power(
    rule: SampledDistributionRule,
    *,
    trials: int,
    corruption_fraction: float,
    minimum_power: float,
    calibration_seed: int,
) -> DistributionPowerCalibration:
    if trials < 20:
        raise ValueError("power calibration requires at least 20 trials")
    if not 0 < corruption_fraction <= 1:
        raise ValueError("power corruption fraction must be in (0, 1]")
    rng = random.Random(calibration_seed)
    corruptions: tuple[
        Literal["first-token-substitution", "early-truncation"], ...
    ] = ("first-token-substitution", "early-truncation")
    detections = {
        "first-token-substitution": 0,
        "early-truncation": 0,
    }
    for trial in range(trials):
        base_arms = tuple(
            _synthetic_arm(
                f"trial-{trial}-arm-{arm_index}",
                rule.samples_per_arm,
                rng,
            )
            for arm_index in range(4)
        )
        for corruption in corruptions:
            candidate_a = _corrupt_arm(
                base_arms[2], corruption, corruption_fraction, rng
            )
            candidate_b = _corrupt_arm(
                base_arms[3], corruption, corruption_fraction, rng
            )
            trial_rule = rule.model_copy(
                update={"bootstrap_seed": rule.bootstrap_seed + trial}
            )
            result = evaluate_sampled_distribution(
                trial_rule,
                target_a=base_arms[0],
                target_b=base_arms[1],
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            )
            detections[corruption] += int(not result.passed)
    return DistributionPowerCalibration(
        minimum_power=minimum_power,
        cases=tuple(
            PowerCalibrationCase(
                corruption=corruption,
                corruption_fraction=corruption_fraction,
                trials=trials,
                detections=detection_count,
            )
            for corruption in corruptions
            for detection_count in (detections[corruption],)
        ),
    )


def _synthetic_arm(
    name: str,
    count: int,
    rng: random.Random,
) -> DistributionArm:
    templates = tuple(
        DistributionSample(
            sampling_seed=rng.randrange(1 << 31),
            output_ids=(token, token + 1, token + 2, token + 3),
            text=f"synthetic-{token}",
            finish_reason="length",
        )
        for token in range(1, 9)
    )
    return DistributionArm(
        arm_id=name,
        server_start_id=f"server-{name}",
        samples=tuple(templates[rng.randrange(len(templates))] for _ in range(count)),
    )


def _corrupt_arm(
    arm: DistributionArm,
    corruption: str,
    fraction: float,
    rng: random.Random,
) -> DistributionArm:
    count = max(1, round(len(arm.samples) * fraction))
    selected = set(rng.sample(range(len(arm.samples)), count))
    samples: list[DistributionSample] = []
    for index, value in enumerate(arm.samples):
        if index not in selected:
            samples.append(value)
        elif corruption == "first-token-substitution":
            samples.append(
                value.model_copy(update={"output_ids": (999_983, *value.output_ids[1:])})
            )
        elif corruption == "early-truncation":
            samples.append(
                value.model_copy(
                    update={
                        "output_ids": value.output_ids[:1],
                        "finish_reason": "stop",
                    }
                )
            )
        else:
            raise ValueError(f"unknown calibration corruption: {corruption}")
    return arm.model_copy(update={"samples": tuple(samples)})


def evaluate_sampled_distribution(
    rule: SampledDistributionRule,
    *,
    target_a: DistributionArm,
    target_b: DistributionArm,
    candidate_a: DistributionArm,
    candidate_b: DistributionArm,
) -> SampledDistributionResult:
    arms = (target_a, target_b, candidate_a, candidate_b)
    starts = (
        target_a.server_start_id,
        target_b.server_start_id,
        candidate_a.server_start_id,
        candidate_b.server_start_id,
    )
    if len(set(starts)) != 4:
        raise ValueError("sampled distribution arms must use four fresh server starts")
    counts = (
        len(target_a.samples),
        len(target_b.samples),
        len(candidate_a.samples),
        len(candidate_b.samples),
    )
    if any(count != rule.samples_per_arm for count in counts):
        raise ValueError("sampled distribution arm count differs from the signed rule")

    target_self = _energy_distance(target_a.samples, target_b.samples)
    candidate_discrepancy = max(
        _energy_distance(target_a.samples, candidate_a.samples),
        _energy_distance(target_b.samples, candidate_b.samples),
    )
    observed_excess = candidate_discrepancy - target_self
    bootstrap = _bootstrap_excess(rule, arms)
    upper = _upper_quantile(bootstrap, rule.confidence_level)
    return SampledDistributionResult(
        rule=rule,
        target_self_discrepancy=target_self,
        target_candidate_discrepancy=candidate_discrepancy,
        observed_excess=observed_excess,
        upper_confidence_bound=upper,
        server_start_ids=starts,
        arm_sample_counts=counts,
    )


def _bootstrap_excess(
    rule: SampledDistributionRule,
    arms: tuple[DistributionArm, DistributionArm, DistributionArm, DistributionArm],
) -> list[float]:
    rng = random.Random(rule.bootstrap_seed)
    matrices = {
        (left, right): _distance_matrix(arms[left].samples, arms[right].samples)
        for left, right in (
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (0, 1),
            (0, 2),
            (1, 3),
        )
    }
    values: list[float] = []
    for _ in range(rule.bootstrap_resamples):
        indices = tuple(
            tuple(rng.randrange(len(arm.samples)) for _ in arm.samples) for arm in arms
        )
        target_self = _indexed_energy_distance(
            indices[0],
            indices[1],
            matrices[(0, 0)],
            matrices[(1, 1)],
            matrices[(0, 1)],
        )
        candidate = max(
            _indexed_energy_distance(
                indices[0],
                indices[2],
                matrices[(0, 0)],
                matrices[(2, 2)],
                matrices[(0, 2)],
            ),
            _indexed_energy_distance(
                indices[1],
                indices[3],
                matrices[(1, 1)],
                matrices[(3, 3)],
                matrices[(1, 3)],
            ),
        )
        values.append(candidate - target_self)
    return values


def _indexed_energy_distance(
    first_indices: Sequence[int],
    second_indices: Sequence[int],
    first_matrix: Sequence[Sequence[float]],
    second_matrix: Sequence[Sequence[float]],
    cross_matrix: Sequence[Sequence[float]],
) -> float:
    cross = _indexed_mean(cross_matrix, first_indices, second_indices, same_group=False)
    within_first = _indexed_mean(
        first_matrix, first_indices, first_indices, same_group=True
    )
    within_second = _indexed_mean(
        second_matrix, second_indices, second_indices, same_group=True
    )
    return max(0.0, (2 * cross) - within_first - within_second)


def _indexed_mean(
    matrix: Sequence[Sequence[float]],
    rows: Sequence[int],
    columns: Sequence[int],
    *,
    same_group: bool,
) -> float:
    total = 0.0
    count = 0
    for row_position, row_index in enumerate(rows):
        start = row_position + 1 if same_group else 0
        for column_index in columns[start:]:
            total += matrix[row_index][column_index]
            count += 1
    return total / count if count else 0.0


def _distance_matrix(
    first: Sequence[DistributionSample],
    second: Sequence[DistributionSample],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(_sample_distance(left, right) for right in second) for left in first
    )


def _energy_distance(
    first: Sequence[DistributionSample],
    second: Sequence[DistributionSample],
) -> float:
    cross = _mean_pair_distance(first, second, same_group=False)
    within_first = _mean_pair_distance(first, first, same_group=True)
    within_second = _mean_pair_distance(second, second, same_group=True)
    return max(0.0, (2 * cross) - within_first - within_second)


def _mean_pair_distance(
    first: Sequence[DistributionSample],
    second: Sequence[DistributionSample],
    *,
    same_group: bool,
) -> float:
    total = 0.0
    count = 0
    for first_index, left in enumerate(first):
        start = first_index + 1 if same_group else 0
        for right in second[start:]:
            total += _sample_distance(left, right)
            count += 1
    return total / count if count else 0.0


def _sample_distance(left: DistributionSample, right: DistributionSample) -> float:
    token_distance = _token_edit_distance(left.output_ids, right.output_ids)
    denominator = max(len(left.output_ids), len(right.output_ids), 1)
    normalized = token_distance / denominator
    if left.finish_reason != right.finish_reason:
        normalized = min(1.0, normalized + 0.1)
    return normalized


def _token_edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_token != right_token),
                )
            )
        previous = current
    return previous[-1]


def _upper_quantile(values: Sequence[float], confidence: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(confidence * len(ordered))))
    return ordered[index]
