"""Copyable pointers to the evidence of a local training run."""

from pathlib import Path

from . import run_observe as observe
from .history import history_source_files
from .paths import clipboard_paths
from .queue import QueueManager
from .runs import RunManager


def ai_context(runs: RunManager, queue: QueueManager, run_id: str, source: str | None) -> dict[str, str]:
    run = runs.detail(run_id)
    attempt = None
    experiment = None
    if run["source"] == "managed":
        attempt = runs.attempt_dir(run_id, run["attemptId"])
        root = attempt.parent.parent
        try:
            experiment = runs.experiment_dir(run_id)
        except FileNotFoundError:
            # An attempt can fail before the trainer creates its experiment folder.
            pass
    else:
        root = experiment = runs.experiment_dir(run_id)

    sources = history_source_files(experiment) if experiment is not None else {}
    if source is not None and source not in sources:
        raise FileNotFoundError("The selected history source does not belong to this run.")
    selected = source if source is not None else next(iter(sources), None)

    lines = [f"traiNNer-GUI run: {run['name']}", f"Run ID: {run_id}", f"Status: {run['status']}"]
    if attempt is not None:
        lines.append(f"Current attempt: {attempt.name}")
    step, total = run.get("iteration"), run.get("total")
    if step is not None:
        progress = f"{step:,}" if total is None else f"{step:,} of {total:,}"
        lines.append(f"Run progress: {progress} iterations")
    if entry_id := run.get("queueEntryId"):
        entry = next((e for e in queue.snapshot()["entries"] if e["id"] == entry_id), {})
        if (parent := entry.get("sourceRunId")) and parent != run_id:
            lines.append(f"Source run: {parent}")
        if entry.get("resumeStep") is not None:
            lines.append(f"Resumed from iteration: {entry['resumeStep']:,}")

    paths = clipboard_paths(root)
    lines.extend(["", f"Run folder on trainer: {paths['native']}"])
    if windows := paths.get("windows"):
        lines.append(f"Same folder on the Windows host: {windows}")
    lines.append("Files below are relative to that folder unless given as absolute paths.")

    def add(label: str, path: Path) -> None:
        if path.is_relative_to(root):
            lines.append(f"{label}: {path.relative_to(root)}")
        else:
            external = clipboard_paths(path)
            lines.append(f"{label}: {external['native']}")
            if windows_path := external.get("windows"):
                lines.append(f"{label} on Windows: {windows_path}")

    if selected is not None:
        label = "Selected TensorBoard events" if selected.startswith("TensorBoard/") else "Selected trainer log"
        add(label, sources[selected])
    else:
        lines.append("Telemetry: no trainer log or TensorBoard event file available.")
    if attempt is not None:
        for label, name in (("Effective recipe (current attempt)", "options.yml"),
                            ("Console log (current attempt)", "console.log"),
                            ("Runtime settings (current attempt)", "runtime.json")):
            if observe.regular_file(attempt / name):
                add(label, attempt / name)
    elif experiment is not None:
        recipe_name = run.get("sourceRecipe")
        recipe = experiment / recipe_name if recipe_name else observe.cli_recipe(experiment)
        if recipe is not None and observe.regular_file(recipe):
            add("Saved recipe", recipe)

    if experiment is not None:
        for label, name in (("Validation images", "visualization"), ("Captured training samples", "training-samples")):
            folder = experiment / name
            if folder.is_dir() and folder.resolve() == folder:
                add(label, folder)
    return {"text": "\n".join(lines)}
