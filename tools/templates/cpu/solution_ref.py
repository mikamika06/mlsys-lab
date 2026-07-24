def traverse(n):
    """Visit all n*n linear indices of a row-major matrix in cache-friendly order."""
    return list(range(n * n))   # row-major = sequential = 1 miss per 64B line
