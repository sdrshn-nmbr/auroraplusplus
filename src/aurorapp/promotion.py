from aurorapp.models import PairedEvaluation, PromotionEligibility, PromotionLimits


def decide_promotion(
    evaluation: PairedEvaluation,
    limits: PromotionLimits,
) -> PromotionEligibility:
    concurrencies = [result.concurrency for result in evaluation.results]
    duplicate_concurrencies = sorted(
        concurrency for concurrency in set(concurrencies) if concurrencies.count(concurrency) > 1
    )
    if duplicate_concurrencies:
        return PromotionEligibility(
            eligible=False,
            reasons=[f"duplicate concurrencies: {duplicate_concurrencies}"],
        )
    by_concurrency = {result.concurrency: result for result in evaluation.results}
    reasons: list[str] = []

    for concurrency in limits.required_concurrencies:
        result = by_concurrency.get(concurrency)
        if result is None:
            reasons.append(f"missing concurrency {concurrency}")
            continue
        prefix = f"concurrency {concurrency}"
        if result.matched_requests < limits.matched_requests_per_arm:
            reasons.append(f"{prefix}: too few matched requests")
        if result.parity_mismatches:
            reasons.append(f"{prefix}: output parity mismatch")
        if result.correctness_regressions:
            reasons.append(f"{prefix}: correctness regression")
        if result.output_rate_lcb_improvement < limits.output_rate_lcb_minimum:
            reasons.append(f"{prefix}: output-rate lower bound below minimum")
        if result.p95_latency_upper_regression > limits.p95_latency_upper_regression_maximum:
            reasons.append(f"{prefix}: p95 latency upper bound regressed")
        if result.p99_latency_upper_regression > limits.p99_latency_upper_regression_maximum:
            reasons.append(f"{prefix}: p99 latency upper bound regressed")
        if result.cold_load_successes < limits.cold_loads_required:
            reasons.append(f"{prefix}: insufficient cold-load successes")
        if result.health_failures:
            reasons.append(f"{prefix}: health failures: {', '.join(result.health_failures)}")

    unexpected = sorted(set(by_concurrency) - set(limits.required_concurrencies))
    if unexpected:
        reasons.append(f"unexpected concurrencies: {unexpected}")
    return PromotionEligibility(eligible=not reasons, reasons=reasons)
