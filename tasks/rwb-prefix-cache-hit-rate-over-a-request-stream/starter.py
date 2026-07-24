def prefix_cache_block_stats(requests: list, block_size: int):
    """
    requests: list of tokenized prompts (list of ints), processed IN
        ORDER against one shared automatic-prefix-cache (vLLM APC style).
    block_size: fixed block size tokens are chunked into.

    Walking each request's blocks left to right, a block is a cache HIT
    (reused) iff every block before it in this request was also a hit
    AND the exact prefix of tokens through the end of this block already
    exists in the cache (from an earlier request). The first
    non-matching block -- and every block after it in this request -- is
    a MISS (computed), since prefix-cache reuse only ever extends an
    exact-matching prefix. A request's own blocks become visible to the
    cache only after the request finishes (a request never reuses its
    own blocks).

    Return (total_reused_blocks, total_computed_blocks) summed over all
    requests.
    """
    raise NotImplementedError('your code here')
