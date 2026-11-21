"""Runtime providers for registered training tasks."""

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from trainner_config.tasks import TRAINING_TASKS, TrainingTaskSpec


def load_task_provider(spec: TrainingTaskSpec) -> ModuleType:
    """Import the selected runtime, including its dataset and architecture registrations."""
    return import_module(spec.model_factory.partition(":")[0])


def load_task_providers() -> None:
    """Load every declared runtime for callers enumerating the complete factory catalog."""
    for spec in TRAINING_TASKS.values():
        load_task_provider(spec)


def discover_task_factory(name: str, available: Callable[[], bool], *, dataset: bool) -> None:
    """Resolve a legacy/direct factory lookup when the caller supplied no owning task."""
    key = name.lower()
    for spec in TRAINING_TASKS.values():
        advertised = spec.dataset_types if dataset else (
            *spec.recommended_architectures, *(spec.architecture_types or ()),
        )
        if any(key == hint.lower() for hint in advertised):
            load_task_provider(spec)
            if available():
                return
    # Architecture recommendations and dataset lists do not enumerate every
    # factory a custom runtime can register as a side effect of its import.
    load_task_providers()
