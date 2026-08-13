from pathlib import Path

from aurorapp.audit import AuditExporter
from aurorapp.factories import FactoryRegistry
from aurorapp.simulation import MemoryEventStore


def test_first_factory_registry_has_only_locked_v1_implementations() -> None:
    registry = FactoryRegistry.v1()

    assert registry.names() == {
        "artifact_store": ("modal-volume",),
        "drafter": ("dflash",),
        "judge": ("codex-app-server",),
        "physical_oracle": ("modal",),
        "serving": ("sglang",),
        "target": ("laguna",),
        "trainer": ("specforge",),
        "workload": ("kernelbook",),
    }


def test_audit_export_is_content_addressed(tmp_path: Path) -> None:
    store = MemoryEventStore()
    exporter = AuditExporter(store)

    first = exporter.export("experiment", tmp_path)
    second = exporter.export("experiment", tmp_path)

    assert first == second
    assert first.is_file()
    assert len(first.parent.name) == 64
