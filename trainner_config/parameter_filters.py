"""Shared name selection for generator optimizer parameter groups."""

import re
from collections.abc import Callable, Sequence


def parameter_filter(include: Sequence[str] | None, exclude: Sequence[str] | None) -> Callable[[str], bool]:
    try:
        included = tuple(re.compile(pattern) for pattern in include or ())
        excluded = tuple(re.compile(pattern) for pattern in exclude or ())
    except re.error as exc:
        raise ValueError(f"Invalid generator optimizer parameter regex: {exc}") from exc

    def selected(name: str) -> bool:
        return (not included or any(pattern.search(name) for pattern in included)) and not any(
            pattern.search(name) for pattern in excluded
        )

    return selected
