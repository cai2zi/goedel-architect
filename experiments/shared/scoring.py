from __future__ import annotations

from typing import Any


def score_sort_key(row: dict[str, Any]) -> tuple[int, float, int, int, int]:
    rollout_id = row.get("rollout_id")
    try:
        rollout_num = int(rollout_id)
    except (TypeError, ValueError):
        rollout_num = 10**9
    return (
        0 if row.get("root_proved") else 1,
        -float(row.get("proved_ratio") or 0.0),
        -int(row.get("proved_node_count") or 0),
        int(row.get("total_nodes") or 0),
        rollout_num,
    )


def pick_best_rollout(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=score_sort_key)[0]


def vote_by_answer(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        answer = row.get("canonical_extracted_answer")
        if answer in (None, ""):
            continue
        clusters.setdefault(str(answer), []).append(row)
    if not clusters:
        return None

    def cluster_key(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, int]:
        _, cluster_rows = item
        min_rollout = min(int(r.get("rollout_id") or 10**9) for r in cluster_rows)
        return (-len(cluster_rows), min_rollout)

    _, winner_rows = sorted(clusters.items(), key=cluster_key)[0]
    return sorted(winner_rows, key=lambda r: int(r.get("rollout_id") or 10**9))[0]


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key)) / len(rows)

