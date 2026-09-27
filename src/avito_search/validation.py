"""Deterministic query-group validation split."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


QUERY_CONTEXT_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]


@dataclass(frozen=True)
class ProxyValidationData:
    train_fold: pl.LazyFrame
    queries: pl.DataFrame
    labels: pl.DataFrame
    statistics: dict[str, int | float]


@dataclass(frozen=True)
class RankerTrainingData:
    queries: pl.DataFrame
    labels: pl.DataFrame


def normalized_query(column: str = "search_query") -> pl.Expr:
    """Normalize casing and whitespace without language-specific assumptions."""
    return (
        pl.col(column)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def split_by_query(
    train: pl.LazyFrame,
    *,
    seed: int = 42,
    fold_count: int = 5,
    fold_index: int = 0,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Keep every normalized query entirely in train or validation."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if not 0 <= fold_index < fold_count:
        raise ValueError("fold_index must be between 0 and fold_count - 1")

    normalized = train.with_columns(normalized_query().alias("query_norm"))
    query_folds = normalized.select("query_norm").unique().with_columns(
        (pl.col("query_norm").hash(seed=seed) % fold_count).alias("fold")
    )
    assigned = normalized.join(query_folds, on="query_norm", how="left")

    train_fold = assigned.filter(pl.col("fold") != fold_index).drop("fold")
    validation_fold = assigned.filter(pl.col("fold") == fold_index).drop("fold")
    return train_fold, validation_fold


def build_proxy_validation(
    train: pl.LazyFrame,
    benchmark_items: pl.LazyFrame,
    *,
    seed: int = 42,
    fold_count: int = 5,
    fold_index: int = 0,
    max_queries: int | None = None,
) -> ProxyValidationData:
    """Build held-out query contexts with labels reachable in the item corpus.

    Splitting uses normalized query text, while evaluation keeps the complete
    search context so locations and filters are not merged together.
    """
    train_fold, validation_fold = split_by_query(
        train,
        seed=seed,
        fold_count=fold_count,
        fold_index=fold_index,
    )

    validation_with_ids = validation_fold.with_columns(
        normalized_query("search_infm_params_text").alias("filter_norm")
    ).with_columns(
        pl.concat_str(
            [
                pl.col("query_norm"),
                pl.col("search_location_id").cast(pl.String),
                pl.col("search_is_delivery_search").cast(pl.String),
                pl.col("filter_norm"),
                pl.col("search_category").cast(pl.String),
            ],
            separator="|",
        )
        .hash(seed=seed)
        .cast(pl.String)
        .alias("query_id")
    )

    all_positive_pairs = validation_with_ids.select("query_id", "item_id").unique()
    reachable_rows = validation_with_ids.join(
        benchmark_items.select("item_id").unique(),
        on="item_id",
        how="inner",
    )
    reachable_labels = reachable_rows.select("query_id", "item_id").unique()
    query_table = reachable_rows.group_by("query_id").agg(
        [pl.col(column).first().alias(column) for column in QUERY_CONTEXT_COLUMNS]
        + [pl.col("query_norm").first().alias("query_norm")]
    )

    queries = query_table.collect().with_columns(
        pl.col("query_id").hash(seed=seed).alias("_sample_order")
    ).sort("_sample_order")
    if max_queries is not None:
        if max_queries <= 0:
            raise ValueError("max_queries must be positive")
        queries = queries.head(max_queries)
    queries = queries.drop("_sample_order")

    labels = (
        reachable_labels
        .join(queries.lazy().select("query_id"), on="query_id", how="inner")
        .collect()
    )

    if queries.get_column("query_id").n_unique() != queries.height:
        raise RuntimeError("A query_id hash collision occurred")
    if labels.is_empty():
        raise ValueError("Proxy validation contains no reachable positive labels")

    all_pair_count = all_positive_pairs.select(pl.len()).collect().item()
    reachable_pair_count = reachable_labels.select(pl.len()).collect().item()
    statistics: dict[str, int | float] = {
        "fold_count": fold_count,
        "fold_index": fold_index,
        "validation_rows": validation_fold.select(pl.len()).collect().item(),
        "all_positive_pairs": all_pair_count,
        "reachable_positive_pairs": reachable_pair_count,
        "reachable_pair_share": round(reachable_pair_count / all_pair_count, 6),
        "evaluated_queries": queries.height,
        "evaluated_positive_pairs": labels.height,
    }

    return ProxyValidationData(
        train_fold=train_fold,
        queries=queries,
        labels=labels,
        statistics=statistics,
    )


def build_ranker_training_data(
    train: pl.LazyFrame,
    benchmark_items: pl.LazyFrame,
    *,
    seed: int = 42,
    max_queries: int = 4000,
) -> RankerTrainingData:
    """Build deterministic train query groups with reachable positive items."""
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")

    rows = train.with_columns(
        normalized_query().alias("query_norm"),
        normalized_query("search_infm_params_text").alias("filter_norm"),
    ).with_columns(
        pl.concat_str(
            [
                pl.col("query_norm"),
                pl.col("search_location_id").cast(pl.String),
                pl.col("search_is_delivery_search").cast(pl.String),
                pl.col("filter_norm"),
                pl.col("search_category").cast(pl.String),
            ],
            separator="|",
        )
        .hash(seed=seed)
        .cast(pl.String)
        .alias("query_id")
    )
    reachable = rows.join(
        benchmark_items.select("item_id").unique(),
        on="item_id",
        how="inner",
    )
    query_table = reachable.group_by("query_id").agg(
        [pl.col(column).first().alias(column) for column in QUERY_CONTEXT_COLUMNS]
    )
    queries = (
        query_table.collect()
        .with_columns(pl.col("query_id").hash(seed=seed).alias("_sample_order"))
        .sort("_sample_order")
        .head(max_queries)
        .drop("_sample_order")
    )
    labels = (
        reachable.select("query_id", "item_id")
        .unique()
        .join(queries.lazy().select("query_id"), on="query_id", how="inner")
        .collect()
    )
    return RankerTrainingData(queries=queries, labels=labels)
