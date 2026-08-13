import pytest

from aurorapp.simulation import DeterministicFaultSimulator, Fault, SimulationInvariantError


@pytest.mark.parametrize(
    "fault",
    [
        Fault.DUPLICATE_TRACE,
        Fault.OUT_OF_ORDER_TRACE,
        Fault.DUPLICATE_TRAINING_JOB,
        Fault.CONTROLLER_DIES_BEFORE_COMMIT,
        Fault.CONTROLLER_DIES_AFTER_COMMIT,
        Fault.STALE_FENCE,
        Fault.TRAINER_OOM,
        Fault.PARTIAL_CHECKPOINT,
        Fault.CORRUPT_CHECKPOINT,
        Fault.JUDGE_OUTAGE,
        Fault.INVALID_SIGNATURE,
        Fault.HARDWARE_SUBSTITUTION,
        Fault.MISSING_METRIC_WINDOW,
        Fault.PUBLIC_DATA_LEAK,
        Fault.DRAIN_DURING_TRANSITION,
    ],
)
def test_fault_does_not_break_safety_invariants(fault: Fault) -> None:
    report = DeterministicFaultSimulator(seed=20260812).run(fault)

    assert report.safety_preserved is True
    assert report.live_resources == ()
    assert report.active_pointer_count <= 1


def test_discrimination_broken_implementation_fails_for_intended_reason() -> None:
    simulator = DeterministicFaultSimulator(seed=17)

    with pytest.raises(SimulationInvariantError, match="public audit data entered training"):
        simulator.run(Fault.PUBLIC_DATA_LEAK, deliberately_broken=True)

    assert simulator.run(Fault.PUBLIC_DATA_LEAK).safety_preserved is True


def test_fresh_unseen_seeds_preserve_invariants() -> None:
    for seed in range(100, 120):
        report = DeterministicFaultSimulator(seed=seed).run(Fault.DRAIN_DURING_TRANSITION)
        assert report.safety_preserved is True
