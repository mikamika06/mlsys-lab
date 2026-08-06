def max_gflops(ai: float, peak_gflops: float, peak_bandwidth: float) -> float:
    ridge = peak_gflops / peak_bandwidth
    if ai < ridge:
        return ai * peak_bandwidth
    return peak_gflops
