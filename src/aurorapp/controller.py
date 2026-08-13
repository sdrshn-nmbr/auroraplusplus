from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from aurorapp.canonical import canonical_sha256
from aurorapp.models import (
    ActivationAuthorization,
    AuthorizedBootstrap,
    AuthorizedPromotion,
    Event,
    EventType,
    HumanApproval,
    SystemMode,
)
from aurorapp.signatures import VerifiedApproval


class RecoveryCertainty(StrEnum):
    CERTAIN = "certain"
    UNCERTAIN = "uncertain"


class CommandConflictError(RuntimeError):
    pass


class Clock(Protocol):
    def monotonic_ns(self) -> int: ...

    def wall_time(self) -> datetime: ...


class EventStore(Protocol):
    events: list[Event]

    def append(self, event: Event) -> Event: ...

    def get_by_idempotency_key(self, key: str) -> Event | None: ...


class Boundaries(Protocol):
    live_resources: list[str]

    def cleanup(self) -> None: ...

    def route_target_only(self) -> None: ...

    def route_drafter(self) -> None: ...

    def route_candidate(self, candidate_id: str) -> None: ...


@dataclass(frozen=True)
class ControllerState:
    mode: SystemMode = SystemMode.OFF
    desired_mode: SystemMode = SystemMode.OFF
    active_drafter: str | None = None
    parent_drafter: str | None = None
    fencing_number: int = 0
    fence_owner: str | None = None


@dataclass(frozen=True)
class CommandResult:
    accepted: bool
    event: Event


def reduce_event(state: ControllerState, event: Event) -> ControllerState:
    values = state.__dict__.copy()
    if event.type is EventType.MODE_REQUESTED:
        values["desired_mode"] = SystemMode(event.payload["mode"])
        values["mode"] = (
            SystemMode(event.payload["mode"])
            if "certainty" in event.payload
            else SystemMode.STARTING
        )
    elif event.type is EventType.RECOVERED_TO_SHADOW:
        values["desired_mode"] = SystemMode.SHADOW
        values["mode"] = SystemMode.SHADOW
    elif event.type is EventType.FENCE_ACQUIRED:
        values["fencing_number"] = event.fencing_number
        values["fence_owner"] = event.payload["owner"]
    elif event.type is EventType.LAST_GOOD_SET:
        values["active_drafter"] = event.payload["drafter"]
        values["parent_drafter"] = event.payload["parent"]
    elif event.type is EventType.CANDIDATE_PROMOTED:
        values["active_drafter"] = event.payload["candidate"]
        values["parent_drafter"] = event.payload["parent"]
    elif event.type is EventType.ROLLED_BACK:
        values["active_drafter"] = event.payload["parent"]
        values["parent_drafter"] = event.payload["grandparent"]
    elif event.type is EventType.LEARNING_FAILED:
        values["mode"] = SystemMode.FAILED
    elif event.type is EventType.DRAIN_STARTED:
        values["mode"] = SystemMode.DRAINING
    elif event.type is EventType.DRAIN_COMPLETED:
        values["mode"] = SystemMode.OFF
        values["desired_mode"] = SystemMode.OFF
    return ControllerState(**values)


class Controller:
    def __init__(
        self,
        experiment_id: str,
        event_store: EventStore,
        boundaries: Boundaries,
        clock: Clock,
    ) -> None:
        self.experiment_id = experiment_id
        self.event_store = event_store
        self.boundaries = boundaries
        self.clock = clock
        self.state = ControllerState()
        for event in self.events:
            if event.experiment_id == experiment_id:
                self.state = reduce_event(self.state, event)

    @property
    def events(self) -> list[Event]:
        return self.event_store.events

    def enable(
        self,
        mode: SystemMode,
        authorization: ActivationAuthorization,
        verified_approval: VerifiedApproval,
        idempotency_key: str,
    ) -> CommandResult:
        if mode not in {SystemMode.SHADOW, SystemMode.AUTO_DRAFTER}:
            raise ValueError("enable mode must be shadow or auto-drafter")
        self._require_verified_approval(authorization.approval, verified_approval)
        if authorization.experiment_id != self.experiment_id:
            raise ValueError("activation experiment ID does not match controller")
        return self._execute(
            EventType.MODE_REQUESTED,
            idempotency_key,
            {
                "mode": mode.value,
                "activation": canonical_sha256(authorization.approval_payload()),
                "compatibility_report": authorization.compatibility_report_hash,
            },
        )

    def recover(self, certainty: RecoveryCertainty, idempotency_key: str) -> CommandResult:
        if certainty is RecoveryCertainty.UNCERTAIN:
            return self._execute(
                EventType.RECOVERED_TO_SHADOW,
                idempotency_key,
                {"certainty": certainty.value},
            )
        mode = self.state.desired_mode
        return self._execute(
            EventType.MODE_REQUESTED,
            idempotency_key,
            {"mode": mode.value, "certainty": certainty.value},
        )

    def acquire_fence(self, owner: str, idempotency_key: str) -> int:
        payload = {"owner": owner, "next_fence": self.state.fencing_number + 1}
        existing = self._matching_existing(EventType.FENCE_ACQUIRED, idempotency_key, payload)
        if existing is not None:
            return existing.fencing_number
        event = self._record(
            EventType.FENCE_ACQUIRED,
            idempotency_key,
            payload,
            fencing_number=self.state.fencing_number + 1,
        )
        return event.fencing_number

    def bootstrap(
        self,
        authorization: AuthorizedBootstrap,
        verified_approval: VerifiedApproval,
        idempotency_key: str,
    ) -> CommandResult:
        self._require_verified_approval(authorization.approval, verified_approval)
        if authorization.experiment_id != self.experiment_id:
            raise ValueError("bootstrap experiment ID does not match controller")
        result = self._execute(
            EventType.LAST_GOOD_SET,
            idempotency_key,
            {
                "drafter": authorization.drafter_id,
                "parent": authorization.parent_id,
                "manifest": authorization.artifact_manifest_hash,
                "activation": authorization.activation_hash,
                "compatibility_report": authorization.compatibility_report_hash,
            },
        )
        if result.accepted:
            self.boundaries.route_drafter()
        return result

    def promote(
        self,
        authorization: AuthorizedPromotion,
        verified_approval: VerifiedApproval,
        fence: int,
        idempotency_key: str,
    ) -> CommandResult:
        self._require_verified_approval(authorization.approval, verified_approval)
        rejection: str | None = None
        if authorization.experiment_id != self.experiment_id:
            rejection = "experiment identity mismatch"
        elif fence != self.state.fencing_number:
            rejection = "stale fencing number"
        elif self.state.active_drafter != authorization.parent_id:
            rejection = "parent is not active"
        elif self.state.mode is not SystemMode.AUTO_DRAFTER:
            rejection = "automatic drafter authority is not active"
        if rejection is not None:
            return self._execute(
                EventType.PROMOTION_REJECTED,
                idempotency_key,
                {"candidate": authorization.candidate_id, "reason": rejection},
                accepted=False,
            )
        result = self._execute(
            EventType.CANDIDATE_PROMOTED,
            idempotency_key,
            {
                "candidate": authorization.candidate_id,
                "parent": authorization.parent_id,
                "physical_report": authorization.physical_report_hash,
                "evaluator_bundle": authorization.evaluator_bundle_hash,
            },
        )
        self.boundaries.route_candidate(authorization.candidate_id)
        return result

    def fail_learning(self, reason: str, idempotency_key: str) -> CommandResult:
        return self._execute(EventType.LEARNING_FAILED, idempotency_key, {"reason": reason})

    def drain(self, idempotency_key: str) -> CommandResult:
        existing = self._matching_existing(EventType.DRAIN_COMPLETED, idempotency_key, {})
        if existing is not None:
            return CommandResult(accepted=True, event=existing)
        self._execute(EventType.DRAIN_STARTED, f"{idempotency_key}:start", {})
        self.boundaries.cleanup()
        if self.boundaries.live_resources:
            self._execute(
                EventType.LEARNING_FAILED,
                f"{idempotency_key}:cleanup-failed",
                {"resources": self.boundaries.live_resources},
            )
            raise RuntimeError(f"resource cleanup incomplete: {self.boundaries.live_resources}")
        return self._execute(EventType.DRAIN_COMPLETED, idempotency_key, {})

    def _execute(
        self,
        event_type: EventType,
        idempotency_key: str,
        payload: dict[str, object],
        accepted: bool = True,
    ) -> CommandResult:
        existing = self._matching_existing(event_type, idempotency_key, payload)
        if existing is not None:
            return CommandResult(accepted=accepted, event=existing)
        event = self._record(event_type, idempotency_key, payload)
        return CommandResult(accepted=accepted, event=event)

    def _matching_existing(
        self,
        event_type: EventType,
        idempotency_key: str,
        payload: dict[str, object],
    ) -> Event | None:
        existing = self.event_store.get_by_idempotency_key(idempotency_key)
        if existing is None:
            return None
        expected_hash = self._command_hash(event_type, payload)
        if existing.command_hash != expected_hash:
            raise CommandConflictError("idempotency key was reused with different command content")
        return existing

    def _record(
        self,
        event_type: EventType,
        idempotency_key: str,
        payload: dict[str, object],
        fencing_number: int | None = None,
    ) -> Event:
        event = Event(
            sequence=len(self.events) + 1,
            experiment_id=self.experiment_id,
            type=event_type,
            idempotency_key=idempotency_key,
            command_hash=self._command_hash(event_type, payload),
            fencing_number=self.state.fencing_number if fencing_number is None else fencing_number,
            monotonic_ns=self.clock.monotonic_ns(),
            wall_time=self.clock.wall_time(),
            payload=payload,
        )
        stored = self.event_store.append(event)
        self.state = reduce_event(self.state, stored)
        return stored

    @staticmethod
    def _command_hash(event_type: EventType, payload: dict[str, object]) -> str:
        return canonical_sha256({"type": event_type.value, "payload": payload})

    @staticmethod
    def _require_verified_approval(
        approval: HumanApproval,
        verified_approval: VerifiedApproval,
    ) -> None:
        if verified_approval.envelope != approval:
            raise ValueError("verified approval does not match authorization envelope")
