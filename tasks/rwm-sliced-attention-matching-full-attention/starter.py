import math

def chunked_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], chunk_size: int) -> tuple[list[list[float]], int]:
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
    raise NotImplementedError('your code here')
