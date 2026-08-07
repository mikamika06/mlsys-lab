def compute_gbps(ms, total_bytes):
    raise NotImplementedError


def roofline_lower_bound_time_ms(total_bytes, peak_bw_gb_s):
    raise NotImplementedError


def find_roofline_knee(block_sizes, gbps_values):
    raise NotImplementedError
