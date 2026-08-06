def compute_tflops(flops, latency_ms):
    sec = latency_ms / 1000.0
    return (flops / sec) / 1e12
