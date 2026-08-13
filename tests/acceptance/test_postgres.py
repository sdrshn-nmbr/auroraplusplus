import os
import uuid

import psycopg
import pytest

from aurorapp.canonical import canonical_sha256
from aurorapp.database import PostgresControlStore, StaleFenceError, UnverifiedCandidateError
from aurorapp.models import ArtifactRef, Event, EventType
from aurorapp.simulation import VirtualClock


@pytest.fixture
def store() -> PostgresControlStore:
    dsn = os.environ.get("AURORAPP_TEST_DATABASE_URL", "postgresql:///postgres")
    schema = f"aurorapp_test_{uuid.uuid4().hex}"
    instance = PostgresControlStore(dsn, schema=schema)
    instance.bootstrap()
    yield instance
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')


def event(key: str, fence: int = 0) -> Event:
    clock = VirtualClock()
    return Event(
        sequence=1,
        experiment_id="experiment",
        type=EventType.MODE_REQUESTED,
        idempotency_key=key,
        command_hash=canonical_sha256({"type": "mode-requested", "payload": {"mode": "shadow"}}),
        fencing_number=fence,
        monotonic_ns=clock.monotonic_ns(),
        wall_time=clock.wall_time(),
        payload={"mode": "shadow"},
    )


def test_command_and_outbox_commit_together(store: PostgresControlStore) -> None:
    first = store.enqueue_command("experiment", "enable-shadow", {"mode": "shadow"}, "same")
    second = store.enqueue_command("experiment", "enable-shadow", {"mode": "shadow"}, "same")

    assert first == second
    assert len(store.pending_outbox()) == 1


def test_events_are_idempotent_and_append_only(store: PostgresControlStore) -> None:
    first = store.append(event("same"))
    second = store.append(event("same"))

    assert first == second
    assert [item.sequence for item in store.load_events("experiment")] == [1]


def test_stale_fence_cannot_change_active_pointer(store: PostgresControlStore) -> None:
    old = store.acquire_fence("experiment", "writer-a")
    store.acquire_fence("experiment", "writer-b")

    with pytest.raises(StaleFenceError):
        store.promote_candidate(
            "experiment",
            candidate_id="candidate",
            expected_parent="parent",
            fence=old,
            idempotency_key="promotion",
        )


def test_unverified_candidate_cannot_become_active(store: PostgresControlStore) -> None:
    fence = store.acquire_fence("experiment", "writer")
    store.set_active_pointer("experiment", "parent", "official", fence)
    store.record_candidate("experiment", "candidate", "parent", "manifest", loader_verified=False)

    with pytest.raises(UnverifiedCandidateError):
        store.promote_candidate(
            "experiment",
            candidate_id="candidate",
            expected_parent="parent",
            fence=fence,
            idempotency_key="promotion",
        )


def test_verified_candidate_promotes_and_rolls_back_atomically(store: PostgresControlStore) -> None:
    fence = store.acquire_fence("experiment", "writer")
    store.set_active_pointer("experiment", "parent", "official", fence)
    store.record_candidate("experiment", "candidate", "parent", "manifest", loader_verified=True)

    store.promote_candidate("experiment", "candidate", "parent", fence, "promote")
    assert store.active_pointer("experiment") == ("candidate", "parent")

    store.rollback("experiment", "candidate", fence, "rollback")
    assert store.active_pointer("experiment") == ("parent", "official")


def test_review_queue_is_fifo_and_decisions_are_immutable(store: PostgresControlStore) -> None:
    first = store.enqueue_review("experiment", {"question": "first"})
    store.enqueue_review("experiment", {"question": "second"})

    assert store.next_review("experiment")["review_id"] == first
    store.answer_review(first, {"answer": "pass", "signature": "signed"})

    assert store.next_review("experiment")["question"] == {"question": "second"}
    with pytest.raises(Exception, match="already answered"):
        store.answer_review(first, {"answer": "fail"})


def test_signed_objects_require_exact_identity_and_one_active_version(
    store: PostgresControlStore,
) -> None:
    store.record_signed_object("a" * 64, "evaluator-bundle", {"version": "a"}, {"sig": "a"})
    store.record_signed_object("b" * 64, "evaluator-bundle", {"version": "b"}, {"sig": "b"})

    store.activate_signed_object("a" * 64, "evaluator-bundle")
    store.activate_signed_object("b" * 64, "evaluator-bundle")

    active = store.active_signed_object("evaluator-bundle")
    assert active is not None
    assert active["object_hash"] == "b" * 64


def test_candidate_inspection_is_scoped_to_experiment(store: PostgresControlStore) -> None:
    store.record_candidate("experiment", "candidate", "parent", "manifest", True)
    store.record_candidate("other", "hidden", "parent", "manifest", True)

    assert [row["candidate_id"] for row in store.list_candidates("experiment")] == ["candidate"]


def test_compatibility_report_and_artifacts_are_control_plane_records(
    store: PostgresControlStore,
) -> None:
    artifact = ArtifactRef(
        content_hash="d" * 64,
        size=7,
        storage_path="modal-volume://objects/dd/value",
        producer="source-probe",
        validation_result="valid",
    )
    store.record_artifact(artifact, {"kind": "source-check"})
    report = {"report_id": "e" * 64, "steps": [{"name": "capture", "status": "failed"}]}
    store.record_compatibility_report("draft", "f" * 64, report, "e" * 64)

    assert store.artifact("d" * 64)["manifest"] == {"kind": "source-check"}
    assert store.compatibility_report("draft", "f" * 64)["report"] == report
