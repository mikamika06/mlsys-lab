def _reused_tokens(requests, block_size):
    cache = set()
    total = 0
    for tokens in requests:
        tokens = list(tokens)
        n = len(tokens)
        num_blocks = (n + block_size - 1) // block_size
        still_reusing = True
        boundary_prefixes = []
        for i in range(num_blocks):
            start = i * block_size
            end = min(start + block_size, n)
            prefix = tuple(tokens[:end])
            boundary_prefixes.append(prefix)
            if still_reusing and prefix in cache:
                total += end - start
            else:
                still_reusing = False
        for prefix in boundary_prefixes:
            cache.add(prefix)
    return total


def block_size_reuse_comparison(requests: list, block_size_a: int, block_size_b: int):
    """
    requests: list of token-id sequences (lists/arrays of ints), processed
    in order against a prefix-caching KV cache (vLLM-style block/radix
    cache): tokens are split into fixed-size blocks, and a block is a
    cache HIT (its tokens are reused, not recomputed) only if the exact
    prefix of tokens up to the end of that block was already produced by
    an earlier request. The first block that doesn't match stops all
    further reuse for that request (prefix caching only ever reuses an
    exact-matching prefix). A request's own blocks become available for
    later requests only after it finishes.

    Simulates two independent caches -- one for block_size_a, one for
    block_size_b -- both fed the SAME `requests` in the same order, and
    returns (reused_tokens_a, reused_tokens_b, better) where `better` is
    "a" if block_size_a reused more tokens in total, "b" if block_size_b
    did, else "tie".
    """
    reused_a = _reused_tokens(requests, block_size_a)
    reused_b = _reused_tokens(requests, block_size_b)
    if reused_a > reused_b:
        better = "a"
    elif reused_b > reused_a:
        better = "b"
    else:
        better = "tie"
    return (reused_a, reused_b, better)
