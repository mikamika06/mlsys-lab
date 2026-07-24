def classify_query(query: str, cache: list[str]) -> str:
    """
    Return 'full', 'partial' or 'miss' depending on how the query string
    matches the cached strings.
    """
    max_lcp = 0
    qlen = len(query)
    for c in cache:
        i = 0
        lc = min(qlen, len(c))
        while i < lc and query[i] == c[i]:
            i += 1
        if i > max_lcp:
            max_lcp = i
            if max_lcp == qlen:   # early exit for full hit
                break
    if max_lcp == qlen:
        return "full"
    if 0 < max_lcp < qlen:
        return "partial"
    return "miss"
