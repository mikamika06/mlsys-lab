import numpy as np


def chunk_counts(prompt_lens: np.ndarray, chunk_budget: int) -> dict:
    """Number of chunked-prefill chunks and the last chunk's size, per prompt.

    Returns {"num_chunks": np.ndarray, "last_chunk": np.ndarray}.
    """
    raise NotImplementedError('your code here')
