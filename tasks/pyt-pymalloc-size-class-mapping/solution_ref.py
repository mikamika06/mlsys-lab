ALIGN = 8
THRESHOLD = 512


def pymalloc_size_class(sizes):
    """pymalloc size-class index: (n-1)//8 for 1<=n<=512, else -1 (raw malloc)."""
    out = []
    for n in sizes:
        if 1 <= n <= THRESHOLD:
            out.append((n - 1) // ALIGN)
        else:
            out.append(-1)
    return out
