"""Retrieval metrics shared by every experiment."""

from __future__ import annotations

import polars as pl


def recall_by_query(
    labels: pl.DataFrame,
    candidates: pl.DataFrame,
    *,
    k: int = 50,
    query_column: str = "query_id",
) -> pl.DataFrame:
    """Calculate Recall@K separately for every labelled query."""
    if k <= 0:
        raise ValueError("k must be positive")

    truth = {
        query: set(item_ids)
        for query, item_ids in (
            labels.select(query_column, "item_id")
            .unique()
            .group_by(query_column)
            .agg(pl.col("item_id"))
            .iter_rows()
        )
    }
    if not truth:
        raise ValueError("labels contain no queries")

    if "rank" in candidates.columns:
        ranked_candidates = candidates.sort([query_column, "rank"])
    else:
        ranked_candidates = candidates.sort(
            [query_column, "score", "item_id"],
            descending=[False, True, False],
        )

    predictions = {
        query: item_ids
        for query, item_ids in (
            ranked_candidates
            .select(query_column, "item_id")
            .unique([query_column, "item_id"], keep="first", maintain_order=True)
            .group_by(query_column, maintain_order=True)
            .agg(pl.col("item_id").head(k))
            .iter_rows()
        )
    }

    rows = [
        {
            query_column: query,
            "relevant_count": len(relevant_items),
            "hit_count": len(relevant_items & set(predictions.get(query, []))),
            "recall": (
                len(relevant_items & set(predictions.get(query, [])))
                / len(relevant_items)
            ),
        }
        for query, relevant_items in truth.items()
    ]
    return pl.DataFrame(rows).sort(query_column)


def recall_at_k(
    labels: pl.DataFrame,
    candidates: pl.DataFrame,
    *,
    k: int = 50,
    query_column: str = "query_id",
) -> float:
    """Calculate macro Recall@K over queries."""
    return recall_by_query(
        labels,
        candidates,
        k=k,
        query_column=query_column,
    ).get_column("recall").mean()
