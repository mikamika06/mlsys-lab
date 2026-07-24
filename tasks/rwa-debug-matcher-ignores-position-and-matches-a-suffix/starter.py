def prefix_reuse_length(cache_tokens, query_tokens):
    # TODO: This incorrectly reports reuse from any matching token subsequence.
    # Prefix reuse requires equality at the same positions.
    if not cache_tokens or not query_tokens:
        return 0

    used = 0
    for token in query_tokens:
        if token in cache_tokens:
            used += 1
        else:
            break
    return used
