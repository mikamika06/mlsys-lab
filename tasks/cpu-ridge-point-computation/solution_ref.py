def ridge_point(peak_flops, peak_bw):
    """Return the roofline ridge point I* = P / B (FLOP / byte)."""
    return peak_flops / peak_bw

def dot_trace(n, l1_bytes, line_bytes):
    """Return a cache-friendly byte-address access trace for a dot product
    of two float64 arrays of length n.

    a lives at addresses 0..8n-1, b at 8n..16n-1.
    Process in chunks that fit in L1 for maximum reuse.
    """
    chunk = l1_bytes // (2 * 8)          # elements of a (or b) per chunk
    trace = []
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        for i in range(lo, hi):          # touch a[i]
            trace.append(i * 8)
        for i in range(lo, hi):          # touch b[i]
            trace.append(8 * n + i * 8)
    return trace
