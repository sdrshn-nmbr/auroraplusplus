from collections.abc import Callable
from typing import Any


class FactoryRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Callable[..., Any]]] = {}

    def register(self, kind: str, name: str, factory: Callable[..., Any]) -> None:
        implementations = self._factories.setdefault(kind, {})
        if name in implementations:
            raise ValueError(f"factory {kind}/{name} is already registered")
        implementations[name] = factory

    def create(self, kind: str, name: str, **configuration: Any) -> Any:
        try:
            factory = self._factories[kind][name]
        except KeyError as error:
            raise KeyError(f"factory {kind}/{name} is not registered") from error
        return factory(**configuration)

    def names(self) -> dict[str, tuple[str, ...]]:
        return {
            kind: tuple(sorted(implementations))
            for kind, implementations in sorted(self._factories.items())
        }

    @classmethod
    def v1(cls) -> "FactoryRegistry":
        registry = cls()
        implementations = {
            "target": "laguna",
            "drafter": "dflash",
            "trainer": "specforge",
            "serving": "sglang",
            "workload": "kernelbook",
            "physical_oracle": "modal",
            "judge": "codex-app-server",
            "artifact_store": "modal-volume",
        }
        for kind, name in implementations.items():
            registry.register(kind, name, lambda **values: values)
        return registry
