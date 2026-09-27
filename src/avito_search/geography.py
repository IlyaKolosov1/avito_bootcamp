"""Simple quota-based fusion of local and global candidates."""

from __future__ import annotations

import polars as pl


CANDIDATE_SCHEMA = {
    "query_id": pl.String,
    "item_id": pl.String,
    "score": pl.Float64,
    "rank": pl.Int64,
    "source": pl.String,
}


def merge_local_and_global(
    query_ids: list[str],
    local_candidates: pl.DataFrame,
    global_candidates: pl.DataFrame,
    *,
    local_limit: int,
    total_limit: int = 50,
) -> pl.DataFrame:
    """Take up to local_limit local items, then fill from global ranking."""
    if local_limit < 0:
        raise ValueError("local_limit cannot be negative")
    if total_limit <= 0:
        raise ValueError("total_limit must be positive")

    local_by_query = _candidate_rows_by_query(local_candidates)
    global_by_query = _candidate_rows_by_query(global_candidates)
    merged_rows: list[dict[str, object]] = []

    for query_id in query_ids:
        selected_items: set[str] = set()
        selected_rows: list[dict[str, object]] = []

        for row in local_by_query.get(query_id, [])[:local_limit]:
            if row["item_id"] not in selected_items:
                selected_items.add(row["item_id"])
                selected_rows.append(row)

        for row in global_by_query.get(query_id, []):
            if len(selected_rows) >= total_limit:
                break
            if row["item_id"] not in selected_items:
                selected_items.add(row["item_id"])
                selected_rows.append(row)

        for final_rank, row in enumerate(selected_rows, start=1):
            merged_rows.append({**row, "rank": final_rank})

    return pl.DataFrame(merged_rows, schema=CANDIDATE_SCHEMA)


def _candidate_rows_by_query(
    candidates: pl.DataFrame,
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in candidates.sort(["query_id", "rank"]).iter_rows(named=True):
        grouped.setdefault(row["query_id"], []).append(row)
    return grouped
