def prefix_reuse_lengths(cache: list[list[int]], queries: list[list[int]], block_size: int) -> list[tuple[int, int]]:
    """For each query, return (exact_reuse, block_reuse):

    exact_reuse: length of the longest prefix `q` shares with any single
        entry in `cache` (RadixAttention-style token-exact reuse).
    block_reuse: exact_reuse rounded down to the nearest multiple of
        block_size (vLLM APC-style whole-block reuse).

    cache: list of previously-seen token-id sequences (lists of int).
    queries: list of new token-id sequences to score against cache.
    block_size: positive int.

    Returns a list of (exact_reuse, block_reuse) int pairs, one per query.
    """
    raise NotImplementedError('your code here')
