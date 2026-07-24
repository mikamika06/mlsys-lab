import numpy as np


def chunk_counts(prompt_lens: np.ndarray, chunk_budget: int) -> dict:
    """Number of chunked-prefill chunks and the last chunk's size, per prompt."""
    n = np.asarray(prompt_lens, dtype=np.int64)
    num_chunks = -(-n // chunk_budget)  # ceil(n / chunk_budget)
    last_chunk = n - (num_chunks - 1) * chunk_budget
    return {"num_chunks": num_chunks, "last_chunk": last_chunk}
