import numpy as np


def streaming_softmax(scores: np.ndarray, chunk_size: int) -> np.ndarray:
    """Compute softmax(scores) by streaming through the vector in chunks.

    scores: 1-D array of raw scores (may be fp32, may contain large or
        very negative values).
    chunk_size: process at most this many elements at a time (the last
        chunk may be smaller if chunk_size doesn't evenly divide len(scores)).

    Maintain a running max `m` and running sum `l` across chunks:
      - m_new = max(m, chunk.max())
      - l = l * exp(m - m_new) + sum(exp(chunk - m_new))
      - m = m_new
    After the last chunk, every output element is exp(x_i - m) / l.

    Returns a float64 array the same shape as `scores`, matching a direct
    numerically-stable softmax over the whole vector.
    """
    raise NotImplementedError('your code here')
