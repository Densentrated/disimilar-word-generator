# cosinesim.py
import numpy as np


def similarity_batch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    x : (B, d)  – B Vietnamese vectors
    y : (N, d)  – N English vectors
    returns : (B, N) cosine similarities
    """
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    return x_norm @ y_norm.T
