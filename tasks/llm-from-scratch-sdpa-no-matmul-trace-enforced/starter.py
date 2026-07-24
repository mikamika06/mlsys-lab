import numpy as np


def sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Single-head scaled dot-product attention, from scratch.

    Q, K, V are float64 arrays of shape (S, d). Return the (S, d) attention
    output. Spell out BOTH contractions (Q.K^T and P.V) with explicit nested
    Python loops -- no @, np.matmul, np.dot, np.einsum, np.tensordot or np.inner.
    Use a max-shifted softmax for numerical stability.
    """
    raise NotImplementedError("your code here")
