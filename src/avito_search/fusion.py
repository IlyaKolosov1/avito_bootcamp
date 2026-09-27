"""Rank fusion for candidates produced by different retrievers."""

from __future__ import annotations

import polars as pl

from avito_search.geography import CANDIDATE_SCHEMA


def reciprocal_rank_fusion(
    query_ids: list[str],
    candidate_tables: list[pl.DataFrame],
    *,
    top_k: int,
    rrf_k: int = 60,
    source: str = "tfidf_rrf",
) -> pl.DataFrame:
    """Fuse rankings using the sum of 1 / (rrf_k + rank)."""
    if not candidate_tables:
        raise ValueError("At least one candidate table is required")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")

    scores_by_query: dict[str, dict[str, float]] = {}
    for candidates in candidate_tables:
        for row in candidates.iter_rows(named=True):
            query_scores = scores_by_query.setdefault(row["query_id"], {})
            query_scores[row["item_id"]] = query_scores.get(row["item_id"], 0.0) + (
                1.0 / (rrf_k + row["rank"])
            )

    fused_rows: list[dict[str, object]] = []
    for query_id in query_ids:
        ranked_items = sorted(
            scores_by_query.get(query_id, {}).items(),
            key=lambda pair: (-pair[1], pair[0]),
        )[:top_k]
        for rank, (item_id, score) in enumerate(ranked_items, start=1):
            fused_rows.append(
                {
                    "query_id": query_id,
                    "item_id": item_id,
                    "score": score,
                    "rank": rank,
                    "source": source,
                }
            )

    return pl.DataFrame(fused_rows, schema=CANDIDATE_SCHEMA)
