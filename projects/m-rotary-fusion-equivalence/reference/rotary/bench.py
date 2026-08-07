def decode_latency_curve(cache_lengths):
    return [0.1 + 0.0005 * L for L in cache_lengths]
