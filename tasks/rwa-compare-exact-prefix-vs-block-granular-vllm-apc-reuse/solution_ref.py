def prefix_reuse_lengths(cache, queries, block_size):
    """For each query, (exact_reuse, block_reuse) against the best-matching
    cache entry: exact_reuse is RadixAttention-style token-exact longest
    common prefix over all of `cache`; block_reuse rounds that down to
    the nearest multiple of block_size (vLLM APC-style)."""
    results = []
    for q in queries:
        best = 0
        for c in cache:
            n = min(len(q), len(c))
            i = 0
            while i < n and q[i] == c[i]:
                i += 1
            if i > best:
                best = i
        block = (best // block_size) * block_size
        results.append((best, block))
    return results
