"""Lazy runtime dispatch for registered standalone task actions."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

from msgspec.structs import replace
from trainner_config.options import ReduxOptions
from trainner_config.tasks import TRAINING_TASKS, TaskAction, require_task_action, task_name

from traiNNer.tasks import load_task_provider


def load_task_action(opt: ReduxOptions, action: TaskAction) -> Callable[..., Any]:
    """Resolve an action before importing its registered runtime module."""
    spec = require_task_action(opt, action)
    if spec.handler is None:
        raise ValueError(f"Task action {action!r} runs through the training model.")
    # Standalone handlers may live outside the module that registers their task's
    # factories. Retain those registrations for providers declaring dataset types.
    provider = TRAINING_TASKS[task_name(opt)]
    if provider.dataset_types:
        load_task_provider(provider)
    module, name = spec.handler.split(":")
    try:
        handler = getattr(import_module(module), name, None)
    except ImportError as exc:
        raise ValueError(
            f"Could not import task action handler {spec.handler!r}. {exc}"
        ) from exc
    if not callable(handler):
        raise ValueError(f"Task action handler {spec.handler!r} must be callable.")
    return handler


def evaluate_holdout(
    opt: ReduxOptions, *, on_dataset: Callable[[dict[str, Any]], None] | None = None
) -> list[dict[str, Any]]:
    """Run the selected provider on prepared validation or test datasets."""
    handler = load_task_action(opt, "holdout")
    reports = handler(replace(opt, is_train=False), on_dataset=on_dataset)
    if not isinstance(reports, list) or any(
        not isinstance(item, dict) for item in reports
    ):
        raise ValueError(
            "The holdout handler must return a list of dataset report objects."
        )
    return reports
