from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from aurorapp.models import ArtifactRef, Event, EventType


class ControlStoreError(RuntimeError):
    pass


class StaleFenceError(ControlStoreError):
    pass


class UnverifiedCandidateError(ControlStoreError):
    pass


class ReviewAlreadyAnsweredError(ControlStoreError):
    pass


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS control_events (
    experiment_id text NOT NULL,
    sequence bigint NOT NULL,
    event_type text NOT NULL,
    idempotency_key text NOT NULL,
    command_hash text NOT NULL,
    fencing_number bigint NOT NULL,
    monotonic_ns bigint NOT NULL,
    wall_time timestamptz NOT NULL,
    payload jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, sequence),
    UNIQUE (experiment_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS commands (
    command_id bigserial PRIMARY KEY,
    experiment_id text NOT NULL,
    command_type text NOT NULL,
    payload jsonb NOT NULL,
    idempotency_key text NOT NULL,
    status text NOT NULL DEFAULT 'pending',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (experiment_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS outbox (
    outbox_id bigserial PRIMARY KEY,
    command_id bigint NOT NULL UNIQUE REFERENCES commands(command_id),
    payload jsonb NOT NULL,
    published_at timestamptz,
    attempts integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS fences (
    experiment_id text PRIMARY KEY,
    fencing_number bigint NOT NULL,
    owner text NOT NULL,
    acquired_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS active_pointers (
    experiment_id text PRIMARY KEY,
    active_drafter text NOT NULL,
    parent_drafter text NOT NULL,
    fencing_number bigint NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS candidates (
    experiment_id text NOT NULL,
    candidate_id text NOT NULL,
    parent_id text NOT NULL,
    manifest_hash text NOT NULL,
    loader_verified boolean NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, candidate_id)
);
CREATE TABLE IF NOT EXISTS promotion_actions (
    experiment_id text NOT NULL,
    idempotency_key text NOT NULL,
    action text NOT NULL,
    candidate_id text NOT NULL,
    active_before text NOT NULL,
    parent_before text NOT NULL,
    fencing_number bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, idempotency_key)
);
CREATE TABLE IF NOT EXISTS artifacts (
    content_hash text PRIMARY KEY,
    size bigint NOT NULL,
    storage_path text NOT NULL,
    producer text NOT NULL,
    validation_result text NOT NULL,
    manifest jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS review_queue (
    review_id bigserial PRIMARY KEY,
    experiment_id text NOT NULL,
    question jsonb NOT NULL,
    status text NOT NULL DEFAULT 'pending',
    decision jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    answered_at timestamptz
);
CREATE TABLE IF NOT EXISTS signed_objects (
    object_hash text PRIMARY KEY,
    object_type text NOT NULL,
    canonical_payload jsonb NOT NULL,
    approval jsonb NOT NULL,
    active boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS compatibility_reports (
    identity_kind text NOT NULL,
    experiment_identity text NOT NULL,
    report_hash text NOT NULL UNIQUE,
    report jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (identity_kind, experiment_identity)
);
CREATE TABLE IF NOT EXISTS worker_leases (
    experiment_id text NOT NULL,
    worker_id text NOT NULL,
    fencing_number bigint NOT NULL,
    expires_at timestamptz NOT NULL,
    cancelled_at timestamptz,
    PRIMARY KEY (experiment_id, worker_id)
);
CREATE TABLE IF NOT EXISTS cursors (
    experiment_id text NOT NULL,
    consumer text NOT NULL,
    cursor_value text NOT NULL,
    fencing_number bigint NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, consumer)
);
"""


class PostgresControlStore:
    def __init__(self, dsn: str, schema: str = "aurorapp") -> None:
        if not schema.replace("_", "").isalnum():
            raise ValueError("schema must contain only letters, numbers, and underscores")
        self.dsn = dsn
        self.schema = schema

    def bootstrap(self) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema))
            )
            connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
            connection.execute(SCHEMA_SQL)

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection[Any]]:
        with psycopg.connect(self.dsn, row_factory=dict_row) as connection:
            connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
            yield connection

    def enqueue_command(
        self,
        experiment_id: str,
        command_type: str,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> int:
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO commands (experiment_id, command_type, payload, idempotency_key)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (experiment_id, idempotency_key) DO NOTHING
                RETURNING command_id
                """,
                (experiment_id, command_type, Jsonb(payload), idempotency_key),
            ).fetchone()
            if row is None:
                row = connection.execute(
                    """
                    SELECT command_id, command_type, payload FROM commands
                    WHERE experiment_id = %s AND idempotency_key = %s
                    """,
                    (experiment_id, idempotency_key),
                ).fetchone()
                if row is None:
                    raise ControlStoreError("idempotent command disappeared")
                if row["command_type"] != command_type or row["payload"] != payload:
                    raise ControlStoreError(
                        "idempotency key conflicts with existing command content"
                    )
            command_id = int(row["command_id"])
            connection.execute(
                """
                INSERT INTO outbox (command_id, payload)
                VALUES (%s, %s)
                ON CONFLICT (command_id) DO NOTHING
                """,
                (command_id, Jsonb({"command_id": command_id, "type": command_type})),
            )
            return command_id

    def pending_outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM outbox WHERE published_at IS NULL ORDER BY outbox_id LIMIT %s",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    @property
    def events(self) -> list[Event]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM control_events ORDER BY experiment_id, sequence"
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def append(self, event: Event) -> Event:
        with self._connection() as connection:
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (event.experiment_id,),
            )
            existing = self._get_event(connection, event.experiment_id, event.idempotency_key)
            if existing is not None:
                return existing
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence
                FROM control_events WHERE experiment_id = %s
                """,
                (event.experiment_id,),
            ).fetchone()
            if row is None:
                raise ControlStoreError("could not allocate event sequence")
            stored = event.model_copy(update={"sequence": int(row["sequence"])})
            connection.execute(
                """
                INSERT INTO control_events (
                    experiment_id, sequence, event_type, idempotency_key,
                    command_hash, fencing_number,
                    monotonic_ns, wall_time, payload
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    stored.experiment_id,
                    stored.sequence,
                    stored.type.value,
                    stored.idempotency_key,
                    stored.command_hash,
                    stored.fencing_number,
                    stored.monotonic_ns,
                    stored.wall_time,
                    Jsonb(stored.payload),
                ),
            )
            return stored

    def get_by_idempotency_key(self, key: str) -> Event | None:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM control_events WHERE idempotency_key = %s ORDER BY created_at",
                (key,),
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 1:
            raise ControlStoreError("idempotency key is ambiguous across experiments")
        return self._event_from_row(rows[0])

    def load_events(self, experiment_id: str) -> list[Event]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM control_events WHERE experiment_id = %s ORDER BY sequence",
                (experiment_id,),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]

    def acquire_fence(self, experiment_id: str, owner: str) -> int:
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO fences (experiment_id, fencing_number, owner)
                VALUES (%s, 1, %s)
                ON CONFLICT (experiment_id) DO UPDATE
                SET fencing_number = fences.fencing_number + 1,
                    owner = EXCLUDED.owner,
                    acquired_at = now()
                RETURNING fencing_number
                """,
                (experiment_id, owner),
            ).fetchone()
            if row is None:
                raise ControlStoreError("failed to acquire fencing number")
            return int(row["fencing_number"])

    def set_active_pointer(
        self,
        experiment_id: str,
        active_drafter: str,
        parent_drafter: str,
        fence: int,
    ) -> None:
        with self._connection() as connection:
            self._require_current_fence(connection, experiment_id, fence)
            connection.execute(
                """
                INSERT INTO active_pointers (
                    experiment_id, active_drafter, parent_drafter, fencing_number
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (experiment_id) DO UPDATE
                SET active_drafter = EXCLUDED.active_drafter,
                    parent_drafter = EXCLUDED.parent_drafter,
                    fencing_number = EXCLUDED.fencing_number,
                    updated_at = now()
                """,
                (experiment_id, active_drafter, parent_drafter, fence),
            )

    def record_candidate(
        self,
        experiment_id: str,
        candidate_id: str,
        parent_id: str,
        manifest_hash: str,
        loader_verified: bool,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO candidates (
                    experiment_id, candidate_id, parent_id, manifest_hash, loader_verified
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (experiment_id, candidate_id) DO NOTHING
                """,
                (experiment_id, candidate_id, parent_id, manifest_hash, loader_verified),
            )

    def promote_candidate(
        self,
        experiment_id: str,
        candidate_id: str,
        expected_parent: str,
        fence: int,
        idempotency_key: str,
    ) -> None:
        with self._connection() as connection:
            self._require_current_fence(connection, experiment_id, fence)
            existing = connection.execute(
                "SELECT 1 FROM promotion_actions WHERE experiment_id = %s AND idempotency_key = %s",
                (experiment_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                return
            candidate = connection.execute(
                """
                SELECT parent_id, loader_verified FROM candidates
                WHERE experiment_id = %s AND candidate_id = %s
                """,
                (experiment_id, candidate_id),
            ).fetchone()
            if candidate is None or not candidate["loader_verified"]:
                raise UnverifiedCandidateError(f"candidate {candidate_id} is not loader-verified")
            if candidate["parent_id"] != expected_parent:
                raise ControlStoreError("candidate parent does not match expected parent")
            pointer = connection.execute(
                """
                SELECT active_drafter, parent_drafter FROM active_pointers
                WHERE experiment_id = %s FOR UPDATE
                """,
                (experiment_id,),
            ).fetchone()
            if pointer is None or pointer["active_drafter"] != expected_parent:
                raise ControlStoreError("expected parent is not active")
            connection.execute(
                """
                INSERT INTO promotion_actions (
                    experiment_id, idempotency_key, action, candidate_id,
                    active_before, parent_before, fencing_number
                ) VALUES (%s, %s, 'promote', %s, %s, %s, %s)
                """,
                (
                    experiment_id,
                    idempotency_key,
                    candidate_id,
                    pointer["active_drafter"],
                    pointer["parent_drafter"],
                    fence,
                ),
            )
            connection.execute(
                """
                UPDATE active_pointers
                SET active_drafter = %s, parent_drafter = %s,
                    fencing_number = %s, updated_at = now()
                WHERE experiment_id = %s
                """,
                (candidate_id, expected_parent, fence, experiment_id),
            )

    def rollback(
        self,
        experiment_id: str,
        failed_candidate: str,
        fence: int,
        idempotency_key: str,
    ) -> None:
        with self._connection() as connection:
            self._require_current_fence(connection, experiment_id, fence)
            existing = connection.execute(
                "SELECT 1 FROM promotion_actions WHERE experiment_id = %s AND idempotency_key = %s",
                (experiment_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                return
            pointer = connection.execute(
                "SELECT active_drafter FROM active_pointers WHERE experiment_id = %s FOR UPDATE",
                (experiment_id,),
            ).fetchone()
            if pointer is None or pointer["active_drafter"] != failed_candidate:
                raise ControlStoreError("failed candidate is not active")
            promotion = connection.execute(
                """
                SELECT active_before, parent_before FROM promotion_actions
                WHERE experiment_id = %s AND candidate_id = %s AND action = 'promote'
                ORDER BY created_at DESC LIMIT 1
                """,
                (experiment_id, failed_candidate),
            ).fetchone()
            if promotion is None:
                raise ControlStoreError("promotion history required for rollback")
            connection.execute(
                """
                INSERT INTO promotion_actions (
                    experiment_id, idempotency_key, action, candidate_id,
                    active_before, parent_before, fencing_number
                ) VALUES (%s, %s, 'rollback', %s, %s, %s, %s)
                """,
                (
                    experiment_id,
                    idempotency_key,
                    failed_candidate,
                    pointer["active_drafter"],
                    promotion["active_before"],
                    fence,
                ),
            )
            connection.execute(
                """
                UPDATE active_pointers
                SET active_drafter = %s, parent_drafter = %s,
                    fencing_number = %s, updated_at = now()
                WHERE experiment_id = %s
                """,
                (
                    promotion["active_before"],
                    promotion["parent_before"],
                    fence,
                    experiment_id,
                ),
            )

    def active_pointer(self, experiment_id: str) -> tuple[str, str] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT active_drafter, parent_drafter FROM active_pointers
                WHERE experiment_id = %s
                """,
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row["active_drafter"]), str(row["parent_drafter"])

    def list_candidates(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT candidate_id, parent_id, manifest_hash, loader_verified, created_at
                FROM candidates WHERE experiment_id = %s ORDER BY created_at, candidate_id
                """,
                (experiment_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_artifact(self, artifact: ArtifactRef, manifest: dict[str, Any]) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO artifacts (
                    content_hash, size, storage_path, producer, validation_result, manifest
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO NOTHING
                """,
                (
                    artifact.content_hash,
                    artifact.size,
                    artifact.storage_path,
                    artifact.producer,
                    artifact.validation_result,
                    Jsonb(manifest),
                ),
            )

    def artifact(self, content_hash: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE content_hash = %s", (content_hash,)
            ).fetchone()
        return None if row is None else dict(row)

    def record_compatibility_report(
        self,
        identity_kind: str,
        experiment_identity: str,
        report: dict[str, Any],
        report_hash: str,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO compatibility_reports (
                    identity_kind, experiment_identity, report_hash, report
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (identity_kind, experiment_identity) DO UPDATE
                SET report_hash = EXCLUDED.report_hash, report = EXCLUDED.report,
                    created_at = now()
                """,
                (identity_kind, experiment_identity, report_hash, Jsonb(report)),
            )

    def compatibility_report(
        self, identity_kind: str, experiment_identity: str
    ) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM compatibility_reports
                WHERE identity_kind = %s AND experiment_identity = %s
                """,
                (identity_kind, experiment_identity),
            ).fetchone()
        return None if row is None else dict(row)

    def enqueue_review(self, experiment_id: str, question: dict[str, Any]) -> int:
        with self._connection() as connection:
            row = connection.execute(
                """
                INSERT INTO review_queue (experiment_id, question)
                VALUES (%s, %s) RETURNING review_id
                """,
                (experiment_id, Jsonb(question)),
            ).fetchone()
        if row is None:
            raise ControlStoreError("review insert returned no identity")
        return int(row["review_id"])

    def next_review(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT review_id, question, created_at FROM review_queue
                WHERE experiment_id = %s AND status = 'pending'
                ORDER BY review_id LIMIT 1
                """,
                (experiment_id,),
            ).fetchone()
        return None if row is None else dict(row)

    def review(self, review_id: int) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM review_queue WHERE review_id = %s", (review_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def answer_review(self, review_id: int, decision: dict[str, Any]) -> None:
        with self._connection() as connection:
            row = connection.execute(
                """
                UPDATE review_queue SET status = 'answered', decision = %s, answered_at = now()
                WHERE review_id = %s AND status = 'pending' RETURNING review_id
                """,
                (Jsonb(decision), review_id),
            ).fetchone()
            if row is None:
                exists = connection.execute(
                    "SELECT status FROM review_queue WHERE review_id = %s", (review_id,)
                ).fetchone()
                if exists is None:
                    raise ControlStoreError(f"review {review_id} does not exist")
                raise ReviewAlreadyAnsweredError(f"review {review_id} is already answered")

    def record_signed_object(
        self,
        object_hash: str,
        object_type: str,
        canonical_payload: dict[str, Any],
        approval: dict[str, Any],
    ) -> None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT object_type, canonical_payload, approval
                FROM signed_objects WHERE object_hash = %s
                """,
                (object_hash,),
            ).fetchone()
            if row is not None:
                if (
                    row["object_type"] != object_type
                    or row["canonical_payload"] != canonical_payload
                    or row["approval"] != approval
                ):
                    raise ControlStoreError("signed object hash conflicts with stored content")
                return
            connection.execute(
                """
                INSERT INTO signed_objects (
                    object_hash, object_type, canonical_payload, approval
                ) VALUES (%s, %s, %s, %s)
                """,
                (object_hash, object_type, Jsonb(canonical_payload), Jsonb(approval)),
            )

    def activate_signed_object(self, object_hash: str, object_type: str) -> None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT object_type FROM signed_objects WHERE object_hash = %s FOR UPDATE
                """,
                (object_hash,),
            ).fetchone()
            if row is None or row["object_type"] != object_type:
                raise ControlStoreError("signed object identity or type mismatch")
            connection.execute(
                "UPDATE signed_objects SET active = false WHERE object_type = %s",
                (object_type,),
            )
            connection.execute(
                "UPDATE signed_objects SET active = true WHERE object_hash = %s",
                (object_hash,),
            )

    def active_signed_object(self, object_type: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT object_hash, canonical_payload, approval, created_at
                FROM signed_objects WHERE object_type = %s AND active
                """,
                (object_type,),
            ).fetchone()
        return None if row is None else dict(row)

    @staticmethod
    def _get_event(
        connection: psycopg.Connection[Any],
        experiment_id: str,
        idempotency_key: str,
    ) -> Event | None:
        row = connection.execute(
            "SELECT * FROM control_events WHERE experiment_id = %s AND idempotency_key = %s",
            (experiment_id, idempotency_key),
        ).fetchone()
        return None if row is None else PostgresControlStore._event_from_row(row)

    @staticmethod
    def _event_from_row(row: dict[str, Any]) -> Event:
        return Event(
            sequence=row["sequence"],
            experiment_id=row["experiment_id"],
            type=EventType(row["event_type"]),
            idempotency_key=row["idempotency_key"],
            command_hash=row["command_hash"],
            fencing_number=row["fencing_number"],
            monotonic_ns=row["monotonic_ns"],
            wall_time=row["wall_time"],
            payload=row["payload"],
        )

    @staticmethod
    def _require_current_fence(
        connection: psycopg.Connection[Any],
        experiment_id: str,
        fence: int,
    ) -> None:
        row = connection.execute(
            "SELECT fencing_number FROM fences WHERE experiment_id = %s FOR UPDATE",
            (experiment_id,),
        ).fetchone()
        if row is None or int(row["fencing_number"]) != fence:
            raise StaleFenceError(f"fencing number {fence} is stale")
