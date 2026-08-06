def compute_mfu(latency_ms, total_flops, peak_tflops_s):
    seconds = latency_ms / 1000.0
    achieved_tflops_s = (total_flops / seconds) / 1e12
    return achieved_tflops_s / peak_tflops_s


def compute_tflops(total_flops, latency_ms):
    seconds = latency_ms / 1000.0
    return (total_flops / seconds) / 1e12
