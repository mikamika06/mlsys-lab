def block_size_reuse_comparison(requests: list, block_size_a: int, block_size_b: int):
    """
    requests: list of token-id sequences (lists/arrays of ints), processed
    in order against a prefix-caching KV cache (vLLM-style block/radix
    cache): tokens are split into fixed-size blocks, and a block is a
    cache HIT (its tokens are reused, not recomputed) only if the exact
    prefix of tokens up to the end of that block was already produced by
    an earlier request. The first block that doesn't match stops all
    further reuse for that request. A request's own blocks become
    available for later requests only after it finishes.

    Simulate two independent caches -- one for block_size_a, one for
    block_size_b -- both fed the SAME `requests` in the same order, and
    return (reused_tokens_a, reused_tokens_b, better) where `better` is
    "a" if block_size_a reused more tokens in total, "b" if block_size_b
    did, else "tie".
    """
    raise NotImplementedError('your code here')
