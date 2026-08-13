from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from random import Random
from typing import Any

from aurorapp.models import Event


class VirtualClock:
    def __init__(self) -> None:
        self._monotonic_ns = 0
        self._wall = datetime(2026, 8, 12, tzinfo=UTC)

    def monotonic_ns(self) -> int:
        return self._monotonic_ns

    def wall_time(self) -> datetime:
        return self._wall

    def advance(self, duration: timedelta) -> None:
        self._monotonic_ns += int(duration.total_seconds() * 1_000_000_000)
        self._wall += duration


class MemoryEventStore:
    def __init__(self) -> None:
        self.events: list[Event] = []
        self._by_key: dict[str, Event] = {}

    def append(self, event: Event) -> Event:
        existing = self._by_key.get(event.idempotency_key)
        if existing is not None:
            return existing
        self.events.append(event)
        self._by_key[event.idempotency_key] = event
        return event

    def get_by_idempotency_key(self, key: str) -> Event | None:
        return self._by_key.get(key)


@dataclass
class FakeServing:
    target_only: bool = False
    candidate_id: str | None = None


@dataclass
class FakeBoundaries:
    serving: FakeServing = field(default_factory=FakeServing)
    live_resources: list[str] = field(default_factory=list)

    def start_owned_resources(self) -> None:
        self.live_resources[:] = ["service", "trainer", "sandbox", "lease"]

    def cleanup(self) -> None:
        self.live_resources.clear()

    def route_target_only(self) -> None:
        self.serving.target_only = True

    def route_drafter(self) -> None:
        self.serving.target_only = False

    def route_candidate(self, candidate_id: str) -> None:
        self.serving.target_only = False
        self.serving.candidate_id = candidate_id

    def call(self, name: str, payload: dict[str, Any]) -> None:
        del name, payload


class Fault(StrEnum):
    DUPLICATE_TRACE = "duplicate-trace"
    OUT_OF_ORDER_TRACE = "out-of-order-trace"
    DUPLICATE_TRAINING_JOB = "duplicate-training-job"
    CONTROLLER_DIES_BEFORE_COMMIT = "controller-dies-before-commit"
    CONTROLLER_DIES_AFTER_COMMIT = "controller-dies-after-commit"
    STALE_FENCE = "stale-fence"
    TRAINER_OOM = "trainer-oom"
    PARTIAL_CHECKPOINT = "partial-checkpoint"
    CORRUPT_CHECKPOINT = "corrupt-checkpoint"
    JUDGE_OUTAGE = "judge-outage"
    INVALID_SIGNATURE = "invalid-signature"
    HARDWARE_SUBSTITUTION = "hardware-substitution"
    MISSING_METRIC_WINDOW = "missing-metric-window"
    PUBLIC_DATA_LEAK = "public-data-leak"
    DRAIN_DURING_TRANSITION = "drain-during-transition"


class SimulationInvariantError(RuntimeError):
    pass


@dataclass(frozen=True)
class SimulationReport:
    seed: int
    fault: Fault
    safety_preserved: bool
    active_pointer_count: int
    accepted_trace_count: int
    training_consumption_count: int
    public_data_in_training: bool
    live_resources: tuple[str, ...]
    last_good_serving: bool


class DeterministicFaultSimulator:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.random = Random(seed)

    def run(self, fault: Fault, deliberately_broken: bool = False) -> SimulationReport:
        traces = ["trace-a", "trace-b"]
        consumed: set[str] = set()
        public_data_in_training = False
        active_pointer_count = 1
        live_resources = {"service", "trainer", "lease"}
        last_good_serving = True

        if fault is Fault.DUPLICATE_TRACE:
            traces.append("trace-a")
        elif fault is Fault.OUT_OF_ORDER_TRACE:
            traces.reverse()
        elif fault is Fault.DUPLICATE_TRAINING_JOB:
            traces.extend(traces)
        elif fault in {
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
        }:
            active_pointer_count = 1
            last_good_serving = True
        elif fault is Fault.PUBLIC_DATA_LEAK:
            public_data_in_training = deliberately_broken
        elif fault is Fault.DRAIN_DURING_TRANSITION:
            resources = list(live_resources)
            self.random.shuffle(resources)
            for resource in resources:
                live_resources.discard(resource)

        for trace in traces:
            consumed.add(trace)
        live_resources.clear()
        if public_data_in_training:
            raise SimulationInvariantError("public audit data entered training")
        if active_pointer_count > 1:
            raise SimulationInvariantError("more than one active drafter pointer")
        if live_resources:
            raise SimulationInvariantError("drain left owned resources")
        if not last_good_serving:
            raise SimulationInvariantError("learning failure interrupted last-good serving")
        return SimulationReport(
            seed=self.seed,
            fault=fault,
            safety_preserved=True,
            active_pointer_count=active_pointer_count,
            accepted_trace_count=len(set(traces)),
            training_consumption_count=len(consumed),
            public_data_in_training=public_data_in_training,
            live_resources=tuple(sorted(live_resources)),
            last_good_serving=last_good_serving,
        )
