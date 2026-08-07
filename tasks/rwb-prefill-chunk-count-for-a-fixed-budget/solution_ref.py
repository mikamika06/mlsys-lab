def chunk_counts(prompt_lens: list[int], chunk_budget: int) -> dict:
    """Number of chunked-prefill chunks and the last chunk's size, per prompt."""
    num_chunks = []
    last_chunk = []
    for n in prompt_lens:
        nc = -(-n // chunk_budget)  # ceil(n / chunk_budget)
        lc = n - (nc - 1) * chunk_budget
        num_chunks.append(nc)
        last_chunk.append(lc)
    return {"num_chunks": num_chunks, "last_chunk": last_chunk}
