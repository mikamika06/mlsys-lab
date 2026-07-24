def edit_distance(source: str, target: str) -> int:
    """Return the Levenshtein edit distance between source and target.

    Uses two-row dynamic programming for O(min(m,n)) space.
    """
    m, n = len(source), len(target)
    if m < n:
        source, target = target, source
        m, n = n, m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        si = source[i - 1]
        for j in range(1, n + 1):
            if si == target[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[n]
