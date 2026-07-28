from __future__ import annotations

from typing import Any


def update_forgetting_metrics(
    current_mean_ce: dict[str, float],
    *,
    first_mean_ce: dict[str, float],
    best_mean_ce: dict[str, float],
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    """Update a lower-is-better continual-learning forgetting matrix row."""
    if not current_mean_ce:
        raise ValueError("Forgetting evaluation requires at least one language")
    if any(not isinstance(value, (int, float)) for value in current_mean_ce.values()):
        raise TypeError("Forgetting CE values must be numeric")

    next_first = dict(first_mean_ce)
    next_best = dict(best_mean_ce)
    rows: dict[str, dict[str, float]] = {}
    for language, raw_value in current_mean_ce.items():
        value = float(raw_value)
        next_first.setdefault(language, value)
        next_best[language] = min(next_best.get(language, value), value)
        rows[language] = {
            "mean_validation_ce": value,
            "first_post_task_mean_validation_ce": next_first[language],
            "best_so_far_mean_validation_ce": next_best[language],
            "ce_change_from_first": value - next_first[language],
            "forgetting_from_best_ce": value - next_best[language],
        }

    count = len(rows)
    summary = {
        "metric": "mean_validation_ce_from_best_v1",
        "lower_is_better": True,
        "language_count": count,
        "languages": rows,
        "average_ce_change_from_first": sum(
            row["ce_change_from_first"] for row in rows.values()
        )
        / count,
        "average_forgetting_from_best_ce": sum(
            row["forgetting_from_best_ce"] for row in rows.values()
        )
        / count,
    }
    return summary, next_first, next_best
