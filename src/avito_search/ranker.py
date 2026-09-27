"""Gradient-boosted reranking over retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from catboost import CatBoostRanker, Pool

from avito_search.geography import CANDIDATE_SCHEMA


COMPONENT_NAMES = [
    "word_tfidf",
    "char_tfidf",
    "word_tfidf_local",
    "char_tfidf_local",
]

FEATURE_COLUMNS = [
    "rrf_score",
    "rrf_rank",
    "word_tfidf_score",
    "word_tfidf_rank",
    "char_tfidf_score",
    "char_tfidf_rank",
    "word_tfidf_local_score",
    "word_tfidf_local_rank",
    "char_tfidf_local_score",
    "char_tfidf_local_rank",
    "location_match",
    "category_match",
    "is_local_candidate",
    "search_is_delivery_search",
    "item_price_log",
    "item_rating",
    "item_reviews_log",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]


@dataclass(frozen=True)
class RankerFitResult:
    model: CatBoostRanker
    statistics: dict[str, int | float]


def build_pair_features(
    candidates: pl.DataFrame,
    components: dict[str, pl.DataFrame],
    queries: pl.DataFrame,
    items: pl.DataFrame,
    *,
    missing_rank: int,
) -> pl.DataFrame:
    """Create one numerical feature row for every query-item candidate pair."""
    features = candidates.select(
        "query_id",
        "item_id",
        pl.col("score").alias("rrf_score"),
        pl.col("rank").alias("rrf_rank"),
        "source",
    )
    for component_name in COMPONENT_NAMES:
        component = components.get(component_name)
        if component is None:
            continue
        features = features.join(
            component.select(
                "query_id",
                "item_id",
                pl.col("score").alias(f"{component_name}_score"),
                pl.col("rank").alias(f"{component_name}_rank"),
            ),
            on=["query_id", "item_id"],
            how="left",
        )

    query_columns = [
        "query_id",
        "search_location_id",
        "search_is_delivery_search",
        "search_category",
    ]
    item_columns = [
        "item_id",
        "item_location_id",
        "item_category_id",
        "item_price",
        "item_rating",
        "item_rating_reviews_count",
        "item_is_phone_hidden",
        "item_is_message_forbidden",
    ]
    features = features.join(
        queries.select(query_columns), on="query_id", how="left"
    ).join(items.select(item_columns), on="item_id", how="left")

    score_columns = [column for column in FEATURE_COLUMNS if column.endswith("_score")]
    rank_columns = [column for column in FEATURE_COLUMNS if column.endswith("_rank")]
    missing_scores = [column for column in score_columns if column not in features.columns]
    missing_ranks = [column for column in rank_columns if column not in features.columns]
    for column in missing_scores:
        features = features.with_columns(pl.lit(0.0).alias(column))
    for column in missing_ranks:
        features = features.with_columns(pl.lit(missing_rank).alias(column))

    return features.with_columns(
        [pl.col(column).fill_null(0.0) for column in score_columns]
        + [pl.col(column).fill_null(missing_rank) for column in rank_columns]
        + [
            (pl.col("search_location_id") == pl.col("item_location_id"))
            .cast(pl.Int8)
            .alias("location_match"),
            (pl.col("search_category") == pl.col("item_category_id"))
            .cast(pl.Int8)
            .alias("category_match"),
            pl.col("source")
            .str.ends_with("_local")
            .cast(pl.Int8)
            .alias("is_local_candidate"),
            pl.col("search_is_delivery_search").fill_null(0).cast(pl.Float64),
            pl.col("item_price")
            .cast(pl.Float64)
            .fill_null(0.0)
            .clip(lower_bound=0.0)
            .log1p()
            .alias("item_price_log"),
            pl.col("item_rating").fill_null(0.0),
            pl.col("item_rating_reviews_count")
            .fill_null(0.0)
            .clip(lower_bound=0.0)
            .log1p()
            .alias("item_reviews_log"),
            pl.col("item_is_phone_hidden").fill_null(False).cast(pl.Int8),
            pl.col("item_is_message_forbidden").fill_null(False).cast(pl.Int8),
        ]
    ).select("query_id", "item_id", *FEATURE_COLUMNS)


def fit_ranker(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    config: dict[str, Any],
) -> RankerFitResult:
    """Train CatBoost on candidate groups that contain a reachable positive."""
    training = (
        features.join(
            labels.with_columns(pl.lit(1).alias("target")),
            on=["query_id", "item_id"],
            how="left",
        )
        .with_columns(pl.col("target").fill_null(0))
        .with_columns(pl.col("target").sum().over("query_id").alias("_positives"))
        .filter(pl.col("_positives") > 0)
        .sort(["query_id", "rrf_rank"])
    )
    if training.is_empty():
        raise ValueError("No positive training candidates were retrieved")

    pool = Pool(
        data=training.select(FEATURE_COLUMNS).to_numpy(),
        label=training.get_column("target").to_numpy(),
        group_id=training.get_column("query_id").to_numpy(),
        feature_names=FEATURE_COLUMNS,
    )
    model = CatBoostRanker(
        iterations=config.get("iterations", 300),
        depth=config.get("depth", 8),
        learning_rate=config.get("learning_rate", 0.08),
        loss_function=config.get("loss_function", "YetiRankPairwise"),
        random_seed=config.get("random_seed", 42),
        task_type=config.get("task_type", "CPU"),
        verbose=config.get("verbose", 50),
        allow_writing_files=False,
    )
    model.fit(pool)

    return RankerFitResult(
        model=model,
        statistics={
            "training_queries": training.get_column("query_id").n_unique(),
            "training_pairs": training.height,
            "training_positives": int(training.get_column("target").sum()),
        },
    )


def rerank_candidates(
    model: CatBoostRanker,
    features: pl.DataFrame,
    *,
    top_k: int = 50,
) -> pl.DataFrame:
    """Score a candidate pool and keep the highest-scoring items per query."""
    scores = np.asarray(model.predict(features.select(FEATURE_COLUMNS).to_numpy()))
    ranked = (
        features.select("query_id", "item_id", "rrf_rank")
        .with_columns(pl.Series("score", scores))
        .sort(
            ["query_id", "score", "rrf_rank", "item_id"],
            descending=[False, True, False, False],
        )
        .group_by("query_id", maintain_order=True)
        .head(top_k)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("query_id").alias("rank"),
            pl.lit("catboost_ranker").alias("source"),
        )
        .select("query_id", "item_id", "score", "rank", "source")
    )
    return pl.DataFrame(ranked, schema=CANDIDATE_SCHEMA)
