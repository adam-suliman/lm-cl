from __future__ import annotations

from typing import Any


def close_iterable(
    iterable: Any,
    *,
    iterator: Any | None = None,
) -> None:
    """Close an iterator and its owner, without closing the same object twice."""
    seen: set[int] = set()
    for candidate in (iterator, iterable):
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        close = getattr(candidate, "close", None)
        if callable(close):
            close()
