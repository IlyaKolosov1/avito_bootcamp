"""Sparse TF-IDF candidate generation."""

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
    analyzer: str = "word"
    max_features: int | None = None


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
    """Exact cosine retrieval over a sparse TF-IDF matrix."""

    def __init__(self, config: TfidfConfig, *, source: str = "word_tfidf") -> None:
        self.config = config
        self.source = source
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer=config.analyzer,
            ngram_range=config.ngram_range,
            min_df=config.min_df,
            max_df=config.max_df,
            max_features=config.max_features,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        self.index: NearestNeighbors | None = None
        self.item_matrix: csr_matrix | None = None
        self.item_ids: np.ndarray | None = None
        self.item_locations: np.ndarray | None = None
        self.location_indices: dict[object, np.ndarray] = {}

    def fit(self, items: pl.DataFrame) -> "TfidfRetriever":
        """Build the vocabulary and exact nearest-neighbour index."""
        if items.get_column("item_id").n_unique() != items.height:
            raise ValueError("item_id must be unique in the retrieval corpus")

        item_texts = combine_text(items, self.config.item_fields)
        item_matrix = self.vectorizer.fit_transform(item_texts).tocsr()

        self.item_ids = items.get_column("item_id").to_numpy()
        self.item_matrix = item_matrix
        if "item_location_id" in items.columns:
            self.item_locations = items.get_column("item_location_id").to_numpy()
            self.location_indices = {
                location: np.flatnonzero(self.item_locations == location)
                for location in np.unique(self.item_locations)
            }
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
                            "source": self.source,
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

    def retrieve_local(
        self,
        queries: pl.DataFrame,
        *,
        query_id_column: str = "query_id",
        query_location_column: str = "search_location_id",
        top_k: int = 40,
    ) -> pl.DataFrame:
        """Retrieve positive-score candidates from the exact query location."""
        if self.item_matrix is None or self.item_ids is None:
            raise RuntimeError("Call fit() before retrieve_local()")
        if self.item_locations is None:
            raise ValueError("Items must include item_location_id for local retrieval")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        required_columns = {query_id_column, query_location_column}
        missing = required_columns - set(queries.columns)
        if missing:
            raise ValueError(f"Missing query columns: {sorted(missing)}")

        query_texts = combine_text(queries, self.config.query_fields)
        query_matrix = self.vectorizer.transform(query_texts).tocsr()
        query_ids = queries.get_column(query_id_column).to_list()
        query_locations = queries.get_column(query_location_column).to_list()

        candidate_rows: list[dict[str, object]] = []
        for row_index, (query_id, location) in enumerate(
            zip(query_ids, query_locations, strict=True)
        ):
            item_indices = self.location_indices.get(location)
            if item_indices is None or len(item_indices) == 0:
                continue

            scores = (
                query_matrix[row_index]
                @ self.item_matrix[item_indices].transpose()
            ).toarray().ravel()
            positive_positions = np.flatnonzero(scores > 0)
            if len(positive_positions) == 0:
                continue

            positive_item_ids = self.item_ids[item_indices[positive_positions]].astype(str)
            order = np.lexsort(
                (
                    positive_item_ids,
                    -scores[positive_positions],
                )
            )[:top_k]

            for rank, ordered_position in enumerate(order, start=1):
                local_position = positive_positions[ordered_position]
                item_index = item_indices[local_position]
                candidate_rows.append(
                    {
                        "query_id": query_id,
                        "item_id": str(self.item_ids[item_index]),
                        "score": float(scores[local_position]),
                        "rank": rank,
                        "source": f"{self.source}_local",
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
