def compute_gbps(ms, total_bytes):
    return float(total_bytes) / (float(ms) * 1e6)


def roofline_lower_bound_time_ms(total_bytes, peak_bw_gb_s):
    return (float(total_bytes) / (float(peak_bw_gb_s) * 1e9)) * 1e3


def find_roofline_knee(block_sizes, gbps_values):
    import numpy as np
    bs = list(block_sizes)
    gv = list(gbps_values)
    max_val = max(gv)
    threshold = 0.95 * max_val
    for b, v in zip(bs, gv):
        if v >= threshold:
            return b
    return bs[np.argmax(gv)]
