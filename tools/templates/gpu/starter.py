def scale_kernel(t, N, a):
    """gmem[i] = a * gmem[i], one thread per element, with COALESCED access.
    Use t.gid, t.gload(idx), t.gstore(idx, val), t.alu(n)."""
    raise NotImplementedError('your code here')
