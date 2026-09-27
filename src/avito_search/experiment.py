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
from avito_search.geography import merge_local_and_global
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


@dataclass(frozen=True)
class ComparisonArtifacts:
    summary: pl.DataFrame
    summary_path: Path


def load_config(path: str | Path) -> dict[str, Any]:
    """Read one immutable experiment configuration."""
    with Path(path).open("rb") as config_file:
        config = tomllib.load(config_file)

    required_sections = {"name", "data", "validation", "tfidf", "geography"}
    missing = required_sections - config.keys()
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")

    return config


def _build_tfidf_config(config: dict[str, Any]) -> TfidfConfig:
    return TfidfConfig(
        item_fields=tuple(config["tfidf"]["item_fields"]),
        query_fields=tuple(config["tfidf"]["query_fields"]),
        ngram_range=tuple(config["tfidf"]["ngram_range"]),
        min_df=config["tfidf"]["min_df"],
    )


def _retrieval_columns(
    config: dict[str, Any],
    tfidf_config: TfidfConfig,
) -> tuple[list[str], list[str]]:
    item_columns = list(dict.fromkeys(["item_id", *tfidf_config.item_fields]))
    query_columns = list(dict.fromkeys(["query_id", *tfidf_config.query_fields]))
    if config["geography"]["enabled"]:
        item_columns.append("item_location_id")
        query_columns.append("search_location_id")
    return item_columns, query_columns


def _retrieve_candidates(
    config: dict[str, Any],
    tfidf_config: TfidfConfig,
    items: pl.DataFrame,
    queries: pl.DataFrame,
) -> tuple[TfidfRetriever, pl.DataFrame]:
    geography = config["geography"]
    total_candidates = (
        geography["local_candidates"] + geography["global_candidates"]
    )
    retriever = TfidfRetriever(tfidf_config).fit(items)

    if not geography["enabled"]:
        return retriever, retriever.retrieve(queries, top_k=total_candidates)

    global_candidates = retriever.retrieve(queries, top_k=total_candidates)
    local_candidates = retriever.retrieve_local(
        queries,
        top_k=geography["local_candidates"],
    )
    candidates = merge_local_and_global(
        queries.get_column("query_id").to_list(),
        local_candidates,
        global_candidates,
        local_limit=geography["local_candidates"],
        total_limit=total_candidates,
    )
    return retriever, candidates


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

    tfidf_config = _build_tfidf_config(config)

    tables = load_dataset(config["data"]["directory"])
    item_columns, query_columns = _retrieval_columns(config, tfidf_config)

    items = tables.benchmark_items.select(item_columns).collect()
    query_source = tables.benchmark_queries.select(query_columns)
    if query_limit is not None:
        query_source = query_source.head(query_limit)
    queries = query_source.collect()

    started_at = time.perf_counter()
    retriever, candidates = _retrieve_candidates(
        config,
        tfidf_config,
        items,
        queries,
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
        "candidates_per_query": (
            config["geography"]["local_candidates"]
            + config["geography"]["global_candidates"]
        ),
        "runtime_seconds": round(runtime_seconds, 3),
        "vocabulary_size": len(retriever.vectorizer.vocabulary_),
        "tfidf": asdict(tfidf_config),
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
    tfidf_config = _build_tfidf_config(config)

    tables = load_dataset(config["data"]["directory"])
    validation = build_proxy_validation(
        tables.train,
        tables.benchmark_items,
        seed=config["seed"],
        fold_count=validation_config["fold_count"],
        fold_index=validation_config["fold_index"],
        max_queries=max_queries,
    )

    item_columns, query_columns = _retrieval_columns(config, tfidf_config)
    items = tables.benchmark_items.select(item_columns).collect()
    queries = validation.queries.select(query_columns)

    started_at = time.perf_counter()
    retriever, candidates = _retrieve_candidates(
        config,
        tfidf_config,
        items,
        queries,
    )
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
        "tfidf": asdict(tfidf_config),
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
