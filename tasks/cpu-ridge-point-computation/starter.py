def ridge_point(peak_flops, peak_bw):
    """Return the roofline ridge point I* = P / B (FLOP / byte)."""
    raise NotImplementedError("your code here")

def dot_trace(n, l1_bytes, line_bytes):
    """Return a cache-friendly byte-address access trace for a dot product
    of two float64 arrays of length n.

    a lives at addresses 0..8n-1, b at 8n..16n-1.
    Process in chunks that fit in L1 for maximum reuse.
    """
    raise NotImplementedError("your code here")
