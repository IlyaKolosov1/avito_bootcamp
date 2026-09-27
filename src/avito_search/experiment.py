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
from avito_search.metrics import recall_at_k, recall_by_query
from avito_search.submission import build_submission, write_submission
from avito_search.tfidf import TfidfConfig, TfidfRetriever
from avito_search.validation import build_proxy_validation


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


def load_config(path: str | Path) -> dict[str, Any]:
    """Read one immutable experiment configuration."""
    with Path(path).open("rb") as config_file:
        config = tomllib.load(config_file)

    required_sections = {"name", "data", "validation", "tfidf", "geography"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    return config


def run_experiment(
    config_path: str | Path,
    *,
    output_dir: str | Path = "runs",
    query_limit: int | None = None,
) -> RunArtifacts:
    """Run the configured benchmark retrieval and save reproducible artifacts."""
    config_path = Path(config_path)
    config = load_config(config_path)
    if config["geography"]["enabled"]:
        raise NotImplementedError(
            "Geographical retrieval belongs to B1 and is not implemented yet"
        )
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be positive")

    tfidf_config = TfidfConfig(
        item_fields=tuple(config["tfidf"]["item_fields"]),
        query_fields=tuple(config["tfidf"]["query_fields"]),
        ngram_range=tuple(config["tfidf"]["ngram_range"]),
        min_df=config["tfidf"]["min_df"],
    )

    tables = load_dataset(config["data"]["directory"])
    item_columns = list(dict.fromkeys(["item_id", *tfidf_config.item_fields]))
    query_columns = list(dict.fromkeys(["query_id", *tfidf_config.query_fields]))

    items = tables.benchmark_items.select(item_columns).collect()
    query_source = tables.benchmark_queries.select(query_columns)
    if query_limit is not None:
        query_source = query_source.head(query_limit)
    queries = query_source.collect()

    started_at = time.perf_counter()
    retriever = TfidfRetriever(tfidf_config).fit(items)
    candidates = retriever.retrieve(
        queries,
        top_k=config["geography"]["global_candidates"],
    )
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

    metadata = {
        "experiment": config["name"],
        "run_type": "submission",
        "query_count": queries.height,
        "item_count": items.height,
        "candidates_per_query": config["geography"]["global_candidates"],
        "runtime_seconds": round(runtime_seconds, 3),
        "vocabulary_size": len(retriever.vectorizer.vocabulary_),
        "tfidf": asdict(tfidf_config),
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
    if config["geography"]["enabled"]:
        raise NotImplementedError(
            "Geographical retrieval belongs to B1 and is not implemented yet"
        )

    validation_config = config["validation"]
    max_queries = query_limit or validation_config.get("max_queries")
    tfidf_config = TfidfConfig(
        item_fields=tuple(config["tfidf"]["item_fields"]),
        query_fields=tuple(config["tfidf"]["query_fields"]),
        ngram_range=tuple(config["tfidf"]["ngram_range"]),
        min_df=config["tfidf"]["min_df"],
    )

    tables = load_dataset(config["data"]["directory"])
    validation = build_proxy_validation(
        tables.train,
        tables.benchmark_items,
        seed=config["seed"],
        fold_count=validation_config["fold_count"],
        fold_index=validation_config["fold_index"],
        max_queries=max_queries,
    )

    item_columns = list(dict.fromkeys(["item_id", *tfidf_config.item_fields]))
    query_columns = list(dict.fromkeys(["query_id", *tfidf_config.query_fields]))
    items = tables.benchmark_items.select(item_columns).collect()
    queries = validation.queries.select(query_columns)

    started_at = time.perf_counter()
    retriever = TfidfRetriever(tfidf_config).fit(items)
    candidates = retriever.retrieve(queries, top_k=50)
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

    metrics = {
        "experiment": config["name"],
        "run_type": "validation",
        "metrics_available": True,
        "query_count": validation.queries.height,
        "positive_pair_count": validation.labels.height,
        **recalls,
        "runtime_seconds": round(runtime_seconds, 3),
        "vocabulary_size": len(retriever.vectorizer.vocabulary_),
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
