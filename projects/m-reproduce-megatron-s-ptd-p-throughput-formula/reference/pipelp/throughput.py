def compute_throughput(pp_size, microbatches, total_flops, time_per_stage):
    bubble_fraction = (pp_size - 1.0) / float(microbatches)
    effective_time = time_per_stage * (1.0 + bubble_fraction)
    tflops_per_sec = total_flops / effective_time / 1e12
    return float(tflops_per_sec)
