def crossover_batch_size(bw_gbps, tflops_w16):
    return float((tflops_w16 * 1000.0) / (2.0 * bw_gbps))
