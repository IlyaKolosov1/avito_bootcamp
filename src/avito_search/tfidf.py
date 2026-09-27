"""Word TF-IDF candidate generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class TfidfConfig:
    item_fields: tuple[str, ...]
    query_fields: tuple[str, ...]
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.995


def combine_text(frame: pl.DataFrame, fields: tuple[str, ...]) -> list[str]:
    """Join selected nullable fields into one normalized text per row."""
    if not fields:
        raise ValueError("At least one text field is required")

    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing text fields: {sorted(missing)}")

    text = frame.select(
        pl.concat_str(
            [pl.col(field).cast(pl.String).fill_null("") for field in fields],
            separator=" ",
        )
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("text")
    )
    return text.get_column("text").to_list()


class TfidfRetriever:
    """Exact cosine retrieval over a sparse word TF-IDF matrix."""

    def __init__(self, config: TfidfConfig) -> None:
        self.config = config
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        self.index: NearestNeighbors | None = None
        self.item_matrix: csr_matrix | None = None
        self.item_ids: np.ndarray | None = None

    def fit(self, items: pl.DataFrame) -> "TfidfRetriever":
        """Build the vocabulary and exact nearest-neighbour index."""
        if items.get_column("item_id").n_unique() != items.height:
            raise ValueError("item_id must be unique in the retrieval corpus")

        item_texts = combine_text(items, self.config.item_fields)
        item_matrix = self.vectorizer.fit_transform(item_texts).tocsr()

        self.item_ids = items.get_column("item_id").to_numpy()
        self.item_matrix = item_matrix
        self.index = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        ).fit(item_matrix)
        return self

    def retrieve(
        self,
        queries: pl.DataFrame,
        *,
        query_id_column: str = "query_id",
        top_k: int = 50,
        batch_size: int = 128,
    ) -> pl.DataFrame:
        """Return ranked candidates without materializing a full score matrix."""
        if self.index is None or self.item_ids is None:
            raise RuntimeError("Call fit() before retrieve()")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if query_id_column not in queries.columns:
            raise ValueError(f"Missing query id column: {query_id_column}")

        query_ids = queries.get_column(query_id_column).to_list()
        query_texts = combine_text(queries, self.config.query_fields)
        query_matrix = self.vectorizer.transform(query_texts).tocsr()
        neighbor_count = min(top_k, len(self.item_ids))

        candidate_rows: list[dict[str, object]] = []
        for batch_start in range(0, queries.height, batch_size):
            batch_stop = min(batch_start + batch_size, queries.height)
            distances, indices = self.index.kneighbors(
                query_matrix[batch_start:batch_stop],
                n_neighbors=neighbor_count,
                return_distance=True,
            )

            for offset, (row_distances, row_indices) in enumerate(
                zip(distances, indices, strict=True)
            ):
                query_id = query_ids[batch_start + offset]
                scores = np.clip(1.0 - row_distances, 0.0, 1.0)
                for rank, (item_index, score) in enumerate(
                    zip(row_indices, scores, strict=True),
                    start=1,
                ):
                    candidate_rows.append(
                        {
                            "query_id": query_id,
                            "item_id": str(self.item_ids[item_index]),
                            "score": float(score),
                            "rank": rank,
                            "source": "word_tfidf",
                        }
                    )

        return pl.DataFrame(
            candidate_rows,
            schema={
                "query_id": pl.String,
                "item_id": pl.String,
                "score": pl.Float64,
                "rank": pl.Int64,
                "source": pl.String,
            },
        )
