import numpy as np


def simulate_decode_metrics(batch_sizes, hidden_dim, weight_bytes, peak_bw, peak_flop_rate):
    throughputs = []
    bandwidths = []
    for b in batch_sizes:
        flops_per_token = 2 * hidden_dim * hidden_dim * b
        mem_bytes_per_token = weight_bytes + b * hidden_dim * 2
        arithmetic_intensity = flops_per_token / mem_bytes_per_token
        occupancy = min(1.0, b / 16.0)
        effective_bw = peak_bw * occupancy
        time_mem = mem_bytes_per_token / effective_bw
        time_compute = flops_per_token / (peak_flop_rate * occupancy)
        total_time = max(time_mem, time_compute)
        tput = b / total_time
        bw_real = mem_bytes_per_token * tput
        throughputs.append(tput)
        bandwidths.append(bw_real)
    return np.array(throughputs), np.array(bandwidths)
