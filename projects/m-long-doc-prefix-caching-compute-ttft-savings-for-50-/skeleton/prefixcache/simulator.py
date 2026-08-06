def compute_block_hashes(tokens: list[int], block_size: int) -> list[int]:
    """Compute deterministic chained prefix hashes for block-aligned sub-sequences."""
    raise NotImplementedError


def simulate_prefix_cache(
    doc_tokens: list[int],
    question_tokens_list: list[list[int]],
    block_size: int,
) -> list[dict]:
    """Simulate sequential requests over a document prefix with block prefix caching."""
    raise NotImplementedError
