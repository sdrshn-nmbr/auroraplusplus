import json
from pathlib import Path
from typing import Protocol

from aurorapp.canonical import canonical_sha256
from aurorapp.models import Event


class ReplayableEventStore(Protocol):
    @property
    def events(self) -> list[Event]: ...


class AuditExporter:
    def __init__(self, event_store: ReplayableEventStore) -> None:
        self.event_store = event_store

    def export(self, experiment_id: str, output_root: Path) -> Path:
        events = [
            event.model_dump(mode="json")
            for event in self.event_store.events
            if event.experiment_id == experiment_id
        ]
        payload = {
            "schema_version": "1",
            "experiment_id": experiment_id,
            "events": events,
        }
        content_hash = canonical_sha256(payload)
        directory = output_root / content_hash
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "audit.json"
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("content-addressed audit export already exists with different bytes")
        if not path.exists():
            path.write_text(encoded, encoding="utf-8")
        return path
