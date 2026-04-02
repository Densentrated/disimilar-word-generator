from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

"""
Analyzes the InputFormat object, and then outputs a dataframe
that contains markers for criteria of a dissimilar word.

This module provides:
- embed_column: embed a text column using a SentenceTransformer model.
- add_cosine_distance: compute row-wise cosine distance between two embedding columns.
"""


# The functions below are used to compute ARBITRARY things, later in the file,
# we will put functions that use these in order to compute SPECIFIC things


def embed_column(
    data: pd.DataFrame,
    col_name: str,
    embedding_col_name: str,
    model: Optional[SentenceTransformer] = None,
) -> pd.DataFrame:
    """Embed the specified text column and add embeddings as a new column.

    Parameters
    - data: DataFrame containing the text column
    - col_name: name of the column to embed (values will be coerced to str)
    - embedding_col_name: name of the output column to store embeddings
    - model: optional SentenceTransformer instance (if None, one will be created)

    Returns
    - A shallow copy of the DataFrame with `embedding_col_name` added. Each cell
      in that column is a numpy.ndarray (dtype float32).
    """
    # Use provided model or create a new one (heavy operation)
    if model is None:
        model = SentenceTransformer("intfloat/multilingual-e5-large")

    # Ensure we have a list of strings; coerce NaN to empty string
    texts: Iterable[str] = [
        ("query: " + str(x)) if pd.notna(x) else "query: "
        for x in data[col_name].tolist()
    ]

    # Compute embeddings (returns numpy array of shape (n, dim))
    embeddings = model.encode(list(texts), batch_size=256, show_progress_bar=True)

    output = data.copy()
    # store numpy arrays per cell for downstream numeric processing
    output[embedding_col_name] = [
        np.asarray(row, dtype=np.float32) for row in embeddings
    ]
    return output


def add_cosine_distance(
    df: pd.DataFrame,
    embedding_col_a: str,
    embedding_col_b: str,
    out_col: str = "cosine_distance",
) -> pd.DataFrame:
    """Compute cosine distance between two embedding columns (row-wise) and append result.

    Cosine distance = 1 - cosine_similarity, where
    cosine_similarity = (a · b) / (||a|| * ||b||).

    Behavior and assumptions:
    - df[embedding_col_a] and df[embedding_col_b] must contain an embedding per row
      (either numpy.ndarray or list-like of numbers).
    - If either vector for a row has zero norm, that row's similarity is treated as 0.0,
      and the distance becomes 1.0.
    - The returned DataFrame is a shallow copy of `df` with `out_col` containing Python floats.

    Raises:
    - ValueError if required columns are missing or embedding dimensionality mismatches.
    """
    if embedding_col_a not in df.columns or embedding_col_b not in df.columns:
        raise ValueError("Embedding columns not found in DataFrame")

    def _to_matrix(series: Iterable) -> np.ndarray:
        """Coerce a sequence of per-row embeddings into a 2D numpy array.

        The parameter is intentionally typed as Iterable to avoid strict Series-only
        typing issues from the type checker (some index/slicing operations may
        produce ambiguous types). Each item must be list-like / ndarray representing
        a 1-D embedding vector.
        """
        rows: List[np.ndarray] = []
        for i, v in enumerate(series):
            if v is None:
                raise ValueError(f"Empty embedding at row {i}")
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(f"Embedding at row {i} is not a 1-D vector")
            rows.append(arr)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        dim = rows[0].shape[0]
        for i, r in enumerate(rows):
            if r.shape[0] != dim:
                raise ValueError(
                    f"Embedding dimensionality mismatch at row {i}: {r.shape[0]} vs {dim}"
                )
        return np.vstack(rows)

    mat_a = _to_matrix(df[embedding_col_a])
    mat_b = _to_matrix(df[embedding_col_b])

    if mat_a.shape != mat_b.shape:
        raise ValueError(
            f"Embedding matrices must have the same shape, got {mat_a.shape} and {mat_b.shape}"
        )

    # Compute dot product per row
    dots = np.einsum("ij,ij->i", mat_a, mat_b)
    norms_a = np.linalg.norm(mat_a, axis=1)
    norms_b = np.linalg.norm(mat_b, axis=1)
    norm_prod = norms_a * norms_b

    sims = np.zeros_like(dots, dtype=np.float32)
    nonzero = norm_prod > 0
    sims[nonzero] = dots[nonzero] / norm_prod[nonzero]

    # Numerical stability: clip to [-1, 1]
    sims = np.clip(sims, -1.0, 1.0)

    distances = 1.0 - sims

    out = df.copy()
    out[out_col] = distances.astype(float).tolist()
    return out


def word_length_of_string_column(
    df: pd.DataFrame, column_name: str, appended_column_name: str
):
    """Appends a column to the dataframe giving the lenght of the strings in that dataframe"""
    if column_name not in df:
        raise ValueError(f"{column_name} not in provided dataframe")

    if appended_column_name in df.columns:
        raise ValueError(
            f"Appended column: {appended_column_name} is already in dataframe"
        )

    out = df.copy()

    series = out[column_name].fillna("").astype(str)

    out[appended_column_name] = series.str.split().apply(len)
    return out

