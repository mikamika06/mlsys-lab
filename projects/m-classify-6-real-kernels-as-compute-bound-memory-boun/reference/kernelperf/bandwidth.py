def compute_achieved_gbps(dram_pct, peak_gbps):
    return (dram_pct / 100.0) * peak_gbps


def cross_check_bandwidth(bytes_sum, time_ns, peak_gbps, tolerance=0.05):
    measured = (bytes_sum / 1e9) / (time_ns / 1e9)
    return measured
