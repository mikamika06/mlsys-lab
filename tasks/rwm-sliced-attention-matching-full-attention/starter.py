import numpy as np

def chunked_attention(Q, K, V, chunk_size):
    """Scaled dot‑product attention with query‑chunking.

    Parameters
    ----------
    Q : ndarray, shape (n_q, d)
        Query matrix.
    K : ndarray, shape (n_k, d)
        Key matrix.
    V : ndarray, shape (n_k, d_v)
        Value matrix.
    chunk_size : int
        Maximum number of queries processed per chunk.

    Returns
    -------
    output : ndarray, shape (n_q, d_v)
        Attention output (identical to full attention).
    peak_bytes : int
        Largest memory (bytes) occupied by any score matrix chunk
        (= chunk_size * n_k * 8).
    """
    raise NotImplementedError("Your implementation here.")
