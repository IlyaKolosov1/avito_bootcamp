"""Loading and validating the three source datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class DatasetTables:
    train: pl.LazyFrame
    benchmark_queries: pl.LazyFrame
    benchmark_items: pl.LazyFrame


def load_dataset(dataset_dir: str | Path = "dataset") -> DatasetTables:
    """Open the project Parquet files lazily and fail clearly if one is missing."""
    dataset_dir = Path(dataset_dir)
    paths = {
        "train": dataset_dir / "train.parquet",
        "benchmark_queries": dataset_dir / "benchmark_queries.parquet",
        "benchmark_items": dataset_dir / "benchmark_items.parquet",
    }

    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")

    return DatasetTables(
        train=pl.scan_parquet(paths["train"]),
        benchmark_queries=pl.scan_parquet(paths["benchmark_queries"]),
        benchmark_items=pl.scan_parquet(paths["benchmark_items"]),
    )
