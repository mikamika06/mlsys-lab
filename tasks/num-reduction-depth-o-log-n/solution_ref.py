def reduction_depth(N):
    """Return ceil(log2(N)) — the tree depth of a parallel reduction of N elements."""
    if N <= 1:
        return 0
    depth = 0
    while N > 1:
        N = (N + 1) // 2
        depth += 1
    return depth
