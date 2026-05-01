from typing import TYPE_CHECKING, Iterable, List, Optional

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

__all__ = ["get_word_count", "embed_string", "orthographic_similarity"]

# Cache for embedding model to avoid reloading
_embedding_model: Optional["SentenceTransformer"] = None
_embedding_model_name: Optional[str] = None


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> "SentenceTransformer":
    """Get or load the sentence embedding model (cached).

    The model is cached globally to avoid reloading it on every call.

    Parameters:
        model_name: Name of the pre-trained sentence-transformers model to use.

    Returns:
        A SentenceTransformer model instance.
    """
    global _embedding_model, _embedding_model_name

    if _embedding_model is None or _embedding_model_name != model_name:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(model_name)
        _embedding_model_name = model_name

    return _embedding_model


def get_word_count(text: str) -> int:
    """Count the number of words in a string.

    Parameters:
        text: Input string to count words in. If None or empty, returns 0.

    Returns:
        The number of words (whitespace-separated tokens).
    """
    if text is None or text == "":
        return 0
    return len(str(text).strip().split())


def embed_string(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """Convert a single string to an embedding vector.

    Uses a pre-trained sentence embedding model to convert text to a vector.
    The model is cached to avoid reloading on every call.

    Parameters:
        text: Input string to embed.
        model_name: Name of the pre-trained sentence-transformers model (defaults to "all-MiniLM-L6-v2").

    Returns:
        Embedding as a list of floats representing the text semantically.
        For None or empty strings, returns a zero vector of the correct dimension.
    """
    model = _get_embedding_model(model_name)
    text_to_embed = "" if text is None else str(text)
    embedding = model.encode(text_to_embed, convert_to_tensor=False)
    return embedding.tolist()


def _to_matrix(series: Iterable) -> np.ndarray:
    """Coerce a sequence of per-row embeddings into a 2D numpy array.

    Each item in `series` must be list-like or an ndarray representing a 1-D
    embedding vector. Raises ValueError for empty embeddings or dimensionality issues.

    Parameters:
        series: Iterable of embedding vectors (each should be 1-D).

    Returns:
        A 2-D numpy array of shape (n_rows, embedding_dim). If the input series
        is empty, returns an array with shape (0, 0).

    Raises:
        ValueError: if any embedding is None, not 1-D, or has mismatched dimensions.
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


def _pairwise_cosine_sims(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """Compute row-wise cosine similarities for two matrices of the same shape.

    Each matrix must have shape (n_rows, embedding_dim) and the same shape as
    the other matrix. The function returns a 1-D numpy array of similarity
    floats in the range [-1, 1].

    Parameters:
        mat_a: First matrix of embeddings (n_rows, embedding_dim).
        mat_b: Second matrix of embeddings (n_rows, embedding_dim).

    Returns:
        1-D numpy array of similarity values, one per row.

    Raises:
        ValueError: if the input matrices do not have the same shape.
    """
    if mat_a.shape != mat_b.shape:
        raise ValueError(
            f"Embedding matrices must have the same shape, got {mat_a.shape} and {mat_b.shape}"
        )

    # Dot product per row
    dots = np.einsum("ij,ij->i", mat_a, mat_b)
    norms_a = np.linalg.norm(mat_a, axis=1)
    norms_b = np.linalg.norm(mat_b, axis=1)
    norm_prod = norms_a * norms_b

    sims = np.zeros_like(dots, dtype=np.float32)
    nonzero = norm_prod > 0
    sims[nonzero] = dots[nonzero] / norm_prod[nonzero]

    # Numerical stability: clip to [-1, 1]
    sims = np.clip(sims, -1.0, 1.0)
    return sims


def orthographic_similarity(word1, word2):
    """
    Returns a similarity score between 0 and 1.
    1 = identical, 0 = completely different.
    Case insensitive.
    """
    word1 = word1.lower()
    word2 = word2.lower()

    m, n = len(word1), len(word2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    distance = dp[m][n]
    max_len = max(m, n)

    if max_len == 0:
        return 1.0

    return 1 - (distance / max_len)

