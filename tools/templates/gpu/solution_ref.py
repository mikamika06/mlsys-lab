def scale_kernel(t, N, a):
    """gmem[i] = a * gmem[i], one thread per element (coalesced)."""
    i = t.gid
    if i < N:
        v = t.gload(i)          # thread i reads element i -> coalesced
        t.alu(1)
        t.gstore(i, a * v)      # writes element i -> coalesced
