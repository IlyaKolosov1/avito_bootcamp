"""Creation and validation of the competition answer file."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def build_submission(
    queries: pl.DataFrame,
    candidates: pl.DataFrame,
    *,
    top_k: int = 50,
) -> pl.DataFrame:
    """Convert ranked candidates to the required query_id/answer format."""
    query_ids = queries.select("query_id").unique(maintain_order=True)
    if query_ids.height != queries.height:
        raise ValueError("query_id must be unique")

    answers = (
        candidates
        .sort(["query_id", "score", "item_id"], descending=[False, True, False])
        .unique(["query_id", "item_id"], keep="first", maintain_order=True)
        .group_by("query_id", maintain_order=True)
        .agg(pl.col("item_id").head(top_k).alias("item_ids"))
        .with_columns(pl.col("item_ids").list.join(" ").alias("answer"))
        .select("query_id", "answer")
    )

    submission = query_ids.join(answers, on="query_id", how="left").with_columns(
        pl.col("answer").fill_null("")
    )
    if submission.height != query_ids.height:
        raise RuntimeError("Submission row count does not match query count")

    return submission


def write_submission(submission: pl.DataFrame, path: str | Path) -> Path:
    """Write a UTF-8 comma-separated answer file without an index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(path)
    return path
