"""Experiment configuration and orchestration."""

from __future__ import annotations

import json
import shutil
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from avito_search.data import load_dataset
from avito_search.fusion import reciprocal_rank_fusion
from avito_search.geography import merge_local_and_global
from avito_search.metrics import recall_at_k, recall_by_query
from avito_search.ranker import build_pair_features, fit_ranker, rerank_candidates
from avito_search.submission import build_submission, write_submission
from avito_search.tfidf import TfidfConfig, TfidfRetriever
from avito_search.validation import (
    build_proxy_validation,
    build_ranker_training_data,
)


@dataclass(frozen=True)
class RunArtifacts:
    run_directory: Path
    predictions_path: Path
    submission_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class EvaluationArtifacts:
    run_directory: Path
    predictions_path: Path
    per_query_metrics_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class ComparisonArtifacts:
    summary: pl.DataFrame
    summary_path: Path


@dataclass(frozen=True)
class RetrievalResult:
    candidates: pl.DataFrame
    components: dict[str, pl.DataFrame]
    vocabulary_sizes: dict[str, int]


def load_config(path: str | Path) -> dict[str, Any]:
    """Read one immutable experiment configuration."""
    with Path(path).open("rb") as config_file:
        config = tomllib.load(config_file)

    required_sections = {"name", "data", "validation", "tfidf", "geography"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    return config


def _build_tfidf_config(
    config: dict[str, Any],
    section: str = "tfidf",
) -> TfidfConfig:
    values = config[section]
    return TfidfConfig(
        item_fields=tuple(values["item_fields"]),
        query_fields=tuple(values["query_fields"]),
        ngram_range=tuple(values["ngram_range"]),
        min_df=values["min_df"],
        analyzer=values.get("analyzer", "word"),
        max_features=values.get("max_features"),
    )


def _build_retriever_configs(
    config: dict[str, Any],
) -> list[tuple[str, TfidfConfig]]:
    retriever_configs = [("word_tfidf", _build_tfidf_config(config))]
    if config.get("char_tfidf", {}).get("enabled", False):
        retriever_configs.append(
            ("char_tfidf", _build_tfidf_config(config, "char_tfidf"))
        )
    return retriever_configs


def _retrieval_columns(
    config: dict[str, Any],
    retriever_configs: list[tuple[str, TfidfConfig]],
) -> tuple[list[str], list[str]]:
    item_fields = [
        field
        for _, tfidf_config in retriever_configs
        for field in tfidf_config.item_fields
    ]
    query_fields = [
        field
        for _, tfidf_config in retriever_configs
        for field in tfidf_config.query_fields
    ]
    item_columns = list(dict.fromkeys(["item_id", *item_fields]))
    query_columns = list(dict.fromkeys(["query_id", *query_fields]))
    if config["geography"]["enabled"]:
        item_columns.append("item_location_id")
        query_columns.append("search_location_id")
    if config.get("ranker", {}).get("enabled", False):
        item_columns.extend(
            [
                "item_category_id",
                "item_price",
                "item_rating",
                "item_rating_reviews_count",
                "item_is_phone_hidden",
                "item_is_message_forbidden",
            ]
        )
        query_columns.extend(["search_is_delivery_search", "search_category"])
    item_columns = list(dict.fromkeys(item_columns))
    query_columns = list(dict.fromkeys(query_columns))
    return item_columns, query_columns


def _fit_retrievers(
    retriever_configs: list[tuple[str, TfidfConfig]],
    items: pl.DataFrame,
) -> list[tuple[str, TfidfRetriever]]:
    return [
        (source, TfidfRetriever(tfidf_config, source=source).fit(items))
        for source, tfidf_config in retriever_configs
    ]


def _retrieve_candidates(
    config: dict[str, Any],
    retrievers: list[tuple[str, TfidfRetriever]],
    queries: pl.DataFrame,
) -> RetrievalResult:
    geography = config["geography"]
    total_candidates = (
        geography["local_candidates"] + geography["global_candidates"]
    )
    fusion = config.get("fusion", {})
    use_fusion = len(retrievers) > 1
    candidate_pool_size = fusion.get("candidate_pool_size", total_candidates)
    if candidate_pool_size < total_candidates:
        raise ValueError(
            "fusion.candidate_pool_size must be at least total candidates"
        )

    query_ids = queries.get_column("query_id").to_list()
    retrieval_limit = candidate_pool_size if use_fusion else total_candidates
    global_tables = [
        retriever.retrieve(queries, top_k=retrieval_limit)
        for _, retriever in retrievers
    ]
    components = {
        source: table
        for (source, _), table in zip(retrievers, global_tables, strict=True)
    }
    global_candidates = global_tables[0]
    if use_fusion:
        global_candidates = reciprocal_rank_fusion(
            query_ids,
            global_tables,
            top_k=total_candidates,
            rrf_k=fusion.get("rrf_k", 60),
            source="tfidf_rrf",
        )

    if not geography["enabled"]:
        candidates = global_candidates
    else:
        local_limit = (
            candidate_pool_size if use_fusion else geography["local_candidates"]
        )
        local_tables = [
            retriever.retrieve_local(queries, top_k=local_limit)
            for _, retriever in retrievers
        ]
        components.update(
            {
                f"{source}_local": table
                for (source, _), table in zip(retrievers, local_tables, strict=True)
            }
        )
        local_candidates = local_tables[0]
        if use_fusion:
            local_candidates = reciprocal_rank_fusion(
                query_ids,
                local_tables,
                top_k=geography["local_candidates"],
                rrf_k=fusion.get("rrf_k", 60),
                source="tfidf_rrf_local",
            )
        candidates = merge_local_and_global(
            query_ids,
            local_candidates,
            global_candidates,
            local_limit=geography["local_candidates"],
            total_limit=total_candidates,
        )

    return RetrievalResult(
        candidates=candidates,
        components=components,
        vocabulary_sizes={
            source: len(retriever.vectorizer.vocabulary_)
            for source, retriever in retrievers
        },
    )


def run_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path = "runs",
    query_limit: int | None = None,
) -> RunArtifacts:
    """Run the configured benchmark retrieval and save reproducible artifacts."""
    config_path = Path(config_path)
    config = load_config(config_path)
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be positive")

    retriever_configs = _build_retriever_configs(config)

    tables = load_dataset(config["data"]["directory"])
    item_columns, query_columns = _retrieval_columns(config, retriever_configs)

    items = tables.benchmark_items.select(item_columns).collect()
    query_source = tables.benchmark_queries.select(query_columns)
    if query_limit is not None:
        query_source = query_source.head(query_limit)
    queries = query_source.collect()

    started_at = time.perf_counter()
    retrievers = _fit_retrievers(retriever_configs, items)
    retrieval = _retrieve_candidates(
        config,
        retrievers,
        queries,
    )
    ranker_result = None
    if config.get("ranker", {}).get("enabled", False):
        ranker_config = config["ranker"]
        training_data = build_ranker_training_data(
            tables.train,
            tables.benchmark_items,
            seed=config["seed"],
            max_queries=ranker_config.get("max_train_queries", 4000),
        )
        training_queries = training_data.queries.select(query_columns)
        training_retrieval = _retrieve_candidates(
            config,
            retrievers,
            training_queries,
        )
        missing_rank = config["fusion"]["candidate_pool_size"] + 1
        training_features = build_pair_features(
            training_retrieval.candidates,
            training_retrieval.components,
            training_queries,
            items,
            missing_rank=missing_rank,
        )
        target_features = build_pair_features(
            retrieval.candidates,
            retrieval.components,
            queries,
            items,
            missing_rank=missing_rank,
        )
        ranker_result = fit_ranker(
            training_features,
            training_data.labels,
            ranker_config,
        )
        candidates = rerank_candidates(
            ranker_result.model,
            target_features,
            top_k=50,
        )
    else:
        candidates = retrieval.candidates
    runtime_seconds = time.perf_counter() - started_at
    submission = build_submission(queries, candidates, top_k=50)

    run_directory = Path(output_dir) / config["name"]
    run_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = run_directory / "predictions.parquet"
    submission_path = run_directory / "answer.csv"
    metadata_path = run_directory / "run.json"

    candidates.write_parquet(predictions_path)
    write_submission(submission, submission_path)
    shutil.copyfile(config_path, run_directory / "config.toml")
    if ranker_result is not None:
        ranker_result.model.save_model(run_directory / "model.cbm")

    metadata = {
        "experiment": config["name"],
        "run_type": "submission",
        "query_count": queries.height,
        "item_count": items.height,
        "candidates_per_query": 50 if ranker_result is not None else (
            config["geography"]["local_candidates"]
            + config["geography"]["global_candidates"]
        ),
        "runtime_seconds": round(runtime_seconds, 3),
        "vocabulary_size": sum(retrieval.vocabulary_sizes.values()),
        "vocabulary_sizes": retrieval.vocabulary_sizes,
        "tfidf": asdict(retriever_configs[0][1]),
        "char_tfidf": config.get("char_tfidf"),
        "fusion": config.get("fusion"),
        "ranker": config.get("ranker"),
        "ranker_statistics": (
            ranker_result.statistics if ranker_result is not None else None
        ),
        "geography": config["geography"],
        "metrics_available": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return RunArtifacts(
        run_directory=run_directory,
        predictions_path=predictions_path,
        submission_path=submission_path,
        metadata_path=metadata_path,
    )


def evaluate_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path = "runs",
    query_limit: int | None = None,
) -> EvaluationArtifacts:
    """Evaluate one experiment on the deterministic proxy-validation set."""
    config_path = Path(config_path)
    config = load_config(config_path)

    validation_config = config["validation"]
    max_queries = query_limit or validation_config.get("max_queries")
    retriever_configs = _build_retriever_configs(config)

    tables = load_dataset(config["data"]["directory"])
    validation = build_proxy_validation(
        tables.train,
        tables.benchmark_items,
        seed=config["seed"],
        fold_count=validation_config["fold_count"],
        fold_index=validation_config["fold_index"],
        max_queries=max_queries,
    )

    item_columns, query_columns = _retrieval_columns(config, retriever_configs)
    items = tables.benchmark_items.select(item_columns).collect()
    queries = validation.queries.select(query_columns)

    started_at = time.perf_counter()
    retrievers = _fit_retrievers(retriever_configs, items)
    retrieval = _retrieve_candidates(
        config,
        retrievers,
        queries,
    )
    ranker_result = None
    if config.get("ranker", {}).get("enabled", False):
        ranker_config = config["ranker"]
        training_data = build_ranker_training_data(
            validation.train_fold,
            tables.benchmark_items,
            seed=config["seed"],
            max_queries=ranker_config.get("max_train_queries", 4000),
        )
        training_queries = training_data.queries.select(query_columns)
        training_retrieval = _retrieve_candidates(
            config,
            retrievers,
            training_queries,
        )
        missing_rank = config["fusion"]["candidate_pool_size"] + 1
        training_features = build_pair_features(
            training_retrieval.candidates,
            training_retrieval.components,
            training_queries,
            items,
            missing_rank=missing_rank,
        )
        validation_features = build_pair_features(
            retrieval.candidates,
            retrieval.components,
            queries,
            items,
            missing_rank=missing_rank,
        )
        ranker_result = fit_ranker(
            training_features,
            training_data.labels,
            ranker_config,
        )
        candidates = rerank_candidates(
            ranker_result.model,
            validation_features,
            top_k=50,
        )
    else:
        candidates = retrieval.candidates
    runtime_seconds = time.perf_counter() - started_at

    recalls = {
        f"recall_at_{k}": recall_at_k(validation.labels, candidates, k=k)
        for k in (10, 20, 50)
    }
    per_query_tables = []
    for k in (10, 20, 50):
        per_query_tables.append(
            recall_by_query(validation.labels, candidates, k=k)
            .select("query_id", pl.col("recall").alias(f"recall_at_{k}"))
        )
    per_query_metrics = per_query_tables[0]
    for table in per_query_tables[1:]:
        per_query_metrics = per_query_metrics.join(table, on="query_id", how="left")

    run_directory = Path(output_dir) / f"{config['name']}_validation"
    run_directory.mkdir(parents=True, exist_ok=True)
    predictions_path = run_directory / "predictions.parquet"
    per_query_metrics_path = run_directory / "per_query_metrics.parquet"
    metrics_path = run_directory / "metrics.json"

    candidates.write_parquet(predictions_path)
    per_query_metrics.write_parquet(per_query_metrics_path)
    shutil.copyfile(config_path, run_directory / "config.toml")
    if ranker_result is not None:
        ranker_result.model.save_model(run_directory / "model.cbm")

    metrics = {
        "experiment": config["name"],
        "run_type": "validation",
        "metrics_available": True,
        "query_count": validation.queries.height,
        "positive_pair_count": validation.labels.height,
        **recalls,
        "runtime_seconds": round(runtime_seconds, 3),
        "vocabulary_size": sum(retrieval.vocabulary_sizes.values()),
        "vocabulary_sizes": retrieval.vocabulary_sizes,
        "tfidf": asdict(retriever_configs[0][1]),
        "char_tfidf": config.get("char_tfidf"),
        "fusion": config.get("fusion"),
        "ranker": config.get("ranker"),
        "ranker_statistics": (
            ranker_result.statistics if ranker_result is not None else None
        ),
        "geography": config["geography"],
        "proxy_validation": validation.statistics,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return EvaluationArtifacts(
        run_directory=run_directory,
        predictions_path=predictions_path,
        per_query_metrics_path=per_query_metrics_path,
        metrics_path=metrics_path,
    )


def compare_evaluations(
    output_dir: str | Path = "runs",
) -> ComparisonArtifacts:
    """Collect validation metrics into one compact comparison table."""
    output_dir = Path(output_dir)
    rows = []
    for metrics_path in sorted(output_dir.glob("*_validation/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment": metrics["experiment"],
                "query_count": metrics["query_count"],
                "recall_at_10": metrics["recall_at_10"],
                "recall_at_20": metrics["recall_at_20"],
                "recall_at_50": metrics["recall_at_50"],
                "runtime_seconds": metrics["runtime_seconds"],
            }
        )

    if not rows:
        raise FileNotFoundError(
            f"No validation metrics found in {output_dir.resolve()}"
        )

    summary = pl.DataFrame(rows).sort("recall_at_50", descending=True)
    summary_path = output_dir / "validation_summary.csv"
    summary.write_csv(summary_path)
    return ComparisonArtifacts(summary=summary, summary_path=summary_path)
