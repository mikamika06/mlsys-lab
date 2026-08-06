def select_optimal_block_size(
    backend: dict, model: dict, candidate_sizes: list, max_memory_bytes: int, prompt_lens: list
) -> dict:
    """Select the optimal block size that fits within budget and minimizes waste."""
    raise NotImplementedError
