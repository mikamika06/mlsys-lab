def chunk_counts(prompt_lens: list[int], chunk_budget: int) -> dict:
    """Number of chunked-prefill chunks and the last chunk's size, per prompt.

    Returns {"num_chunks": list[float], "last_chunk": list[float]}.
    """
    raise NotImplementedError('your code here')
