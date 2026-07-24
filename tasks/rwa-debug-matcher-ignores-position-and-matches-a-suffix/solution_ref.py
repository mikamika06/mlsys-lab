def prefix_reuse_length(cache_tokens, query_tokens):
    length = 0
    limit = min(len(cache_tokens), len(query_tokens))
    while length < limit and cache_tokens[length] == query_tokens[length]:
        length += 1
    return length
