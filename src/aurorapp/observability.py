import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider


class MachineEvidenceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, identity: Mapping[str, str], values: Mapping[str, Any]) -> None:
        required = {
            "experiment_id",
            "worker_id",
            "model_hash",
            "evaluator_bundle_hash",
            "data_manifest_hash",
            "hardware_identity",
            "fencing_number",
        }
        missing = required - set(identity)
        if missing:
            raise ValueError(f"machine evidence is missing identity fields: {sorted(missing)}")
        payload = {
            "event": event,
            "identity": dict(identity),
            "monotonic_ns": time.monotonic_ns(),
            "wall_time_ns": time.time_ns(),
            "values": dict(values),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def configure_observability() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    trace.set_tracer_provider(TracerProvider())
    metrics.set_meter_provider(MeterProvider())
