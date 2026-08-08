from draftquant.metrics import compute_draft_latency, compute_speculative_throughput


def find_int8_threshold(draft_sizes, target_latency, gamma, acceptance_rate, int8_dequant_overhead_us, bandwidth_gbps, launch_overhead_us):
    """Find parameter size below which FP16 draft model outperforms INT8 draft model."""
    crossover_size = None
    for size in sorted(draft_sizes):
        fp16_lat = compute_draft_latency(size, "fp16", bandwidth_gbps, launch_overhead_us)
        int8_lat = compute_draft_latency(size, "int8", bandwidth_gbps, launch_overhead_us) + int8_dequant_overhead_us

        fp16_tp = compute_speculative_throughput(target_latency, fp16_lat, gamma, acceptance_rate)
        int8_tp = compute_speculative_throughput(target_latency, int8_lat, gamma, acceptance_rate)

        if int8_tp > fp16_tp:
            crossover_size = size
            break

    return crossover_size
