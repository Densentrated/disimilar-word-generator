from typing import Optional

import pandas as pd

from cognate_similarity_processor import compute_cognate_similarity
from words_processor import (
    _pairwise_cosine_sims,
    _to_matrix,
    embed_string,
    get_word_count,
    orthographic_similarity,
)

__all__ = [
    "add_cosine_distance",
    "add_word_count_column",
    "embed_column",
    "add_orthographic_similarity",
    "add_cognate_similarity",
]


def add_cosine_distance(
    df: pd.DataFrame,
    embedding_col_a: str,
    embedding_col_b: str,
    out_col: str = "cosine_distance",
) -> pd.DataFrame:
    """Compute cosine distance between two embedding columns (row-wise) and append result.

    Uses the word_processor functions to convert embeddings to matrices and compute
    row-wise cosine similarities. Cosine distance = 1 - cosine_similarity.

    Parameters:
        df: DataFrame that contains the embedding columns.
        embedding_col_a: Column name for the first embedding per row.
        embedding_col_b: Column name for the second embedding per row.
        out_col: Name of the output column to store distances (defaults to "cosine_distance").

    Returns:
        A shallow copy of `df` with `out_col` containing Python floats (one per row).

    Raises:
        ValueError: if either embedding column is missing.
        ValueError: if embeddings are malformed (non-1D, mismatched dims, etc.).
    """
    if embedding_col_a not in df.columns or embedding_col_b not in df.columns:
        raise ValueError("Embedding columns not found in DataFrame")

    mat_a = _to_matrix(df[embedding_col_a])
    mat_b = _to_matrix(df[embedding_col_b])

    sims = _pairwise_cosine_sims(mat_a, mat_b)
    distances = 1.0 - sims

    out = df.copy()
    out[out_col] = distances.astype(float).tolist()
    return out


def add_word_count_column(
    df: pd.DataFrame,
    text_column: str,
    output_column: Optional[str] = None,
) -> pd.DataFrame:
    """Add a column containing word counts for text in a specified column.

    Uses the word_processor.get_word_count function to count words in each row.

    Parameters:
        df: Input DataFrame.
        text_column: Name of the column containing text to count words from.
        output_column: Name of the output column (defaults to f"{text_column}_word_count").

    Returns:
        A shallow copy of `df` with the word count column appended.

    Raises:
        ValueError: if text_column is not in the DataFrame.
        ValueError: if output_column already exists in the DataFrame.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    if output_column is None:
        output_column = f"{text_column}_word_count"

    if output_column in df.columns:
        raise ValueError(f"Output column '{output_column}' already exists in DataFrame")

    out = df.copy()
    out[output_column] = out[text_column].fillna("").astype(str).apply(get_word_count)
    return out


def embed_column(
    df: pd.DataFrame,
    text_column: str,
    output_column: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """Embed text in a column using a pre-trained transformer model.

    Applies the word_processor.embed_string function to each row to generate
    embeddings for text in the specified column.

    Parameters:
        df: Input DataFrame.
        text_column: Name of the column containing text to embed.
        output_column: Name of the output column to store embeddings (as lists).
        model_name: Name of the pre-trained sentence-transformers model (defaults to "all-MiniLM-L6-v2").

    Returns:
        A shallow copy of `df` with the embedding column appended.

    Raises:
        ValueError: if text_column is not in the DataFrame.
        ValueError: if output_column already exists in the DataFrame.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    if output_column in df.columns:
        raise ValueError(f"Output column '{output_column}' already exists in DataFrame")

    out = df.copy()
    out[output_column] = (
        out[text_column]
        .fillna("")
        .astype(str)
        .apply(lambda text: embed_string(text, model_name=model_name))
    )
    return out


def add_orthographic_similarity(
    df: pd.DataFrame,
    text_col_a: str,
    text_col_b: str,
    output_column: Optional[str] = None,
) -> pd.DataFrame:
    """Compute orthographic similarity between two text columns (row-wise) and append result.

    Uses the word_processor.orthographic_similarity function (Levenshtein distance based)
    to measure character-level similarity between text strings in two columns.

    Orthographic similarity ranges from 0 (completely different) to 1 (identical).

    Parameters:
        df: Input DataFrame.
        text_col_a: Name of the first text column.
        text_col_b: Name of the second text column.
        output_column: Name of the output column (defaults to "orthographic_similarity").

    Returns:
        A shallow copy of `df` with the similarity column appended.

    Raises:
        ValueError: if either text column is not in the DataFrame.
        ValueError: if output_column already exists in the DataFrame.
    """
    if text_col_a not in df.columns or text_col_b not in df.columns:
        raise ValueError(f"Text columns not found in DataFrame")

    if output_column is None:
        output_column = "orthographic_similarity"

    if output_column in df.columns:
        raise ValueError(f"Output column '{output_column}' already exists in DataFrame")

    out = df.copy()
    out[output_column] = out.apply(
        lambda row: orthographic_similarity(
            str(row[text_col_a]).lower() if row[text_col_a] else "",
            str(row[text_col_b]).lower() if row[text_col_b] else "",
        ),
        axis=1,
    )
    return out


def add_cognate_similarity(
    df: pd.DataFrame,
    ipa_col_a: str,
    ipa_col_b: str,
    output_column: Optional[str] = None,
    sound_class_model: str = "sca",
) -> pd.DataFrame:
    """Compute cognate similarity between two IPA columns (row-wise) and append result.

    Uses the cognate_similarity_processor.compute_cognate_similarity function to measure
    phonological similarity between words in IPA notation using linguistically informed
    sound class alignment.

    Cognate similarity ranges from 0 (completely different) to 1 (identical).

    Parameters:
        df: Input DataFrame.
        ipa_col_a: Name of the first IPA text column.
        ipa_col_b: Name of the second IPA text column.
        output_column: Name of the output column (defaults to "cognate_similarity").
        sound_class_model: Sound class model to use for alignment (defaults to "sca").
                          Other options: "dolgo", "asjp".

    Returns:
        A shallow copy of `df` with the cognate similarity column appended.

    Raises:
        ValueError: if either IPA column is not in the DataFrame.
        ValueError: if output_column already exists in the DataFrame.
    """
    if ipa_col_a not in df.columns or ipa_col_b not in df.columns:
        raise ValueError(f"IPA columns not found in DataFrame")

    if output_column is None:
        output_column = "cognate_similarity"

    if output_column in df.columns:
        raise ValueError(f"Output column '{output_column}' already exists in DataFrame")

    out = df.copy()
    out[output_column] = out.apply(
        lambda row: compute_cognate_similarity(
            str(row[ipa_col_a]) if row[ipa_col_a] else "",
            str(row[ipa_col_b]) if row[ipa_col_b] else "",
            sound_class_model=sound_class_model,
        ),
        axis=1,
    )
    return out
