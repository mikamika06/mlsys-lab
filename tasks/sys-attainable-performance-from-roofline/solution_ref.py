import numpy as np

def roofline_perf(ai, peak_flops, mem_bandwidth):
    """Return attainable performance per configuration."""
    ai = np.asarray(ai, dtype=np.float64)
    peak = np.asarray(peak_flops, dtype=np.float64)
    bw = np.asarray(mem_bandwidth, dtype=np.float64)
    return np.minimum(peak, ai * bw)
