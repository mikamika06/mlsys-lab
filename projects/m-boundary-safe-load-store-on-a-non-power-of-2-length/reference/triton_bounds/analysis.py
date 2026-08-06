import math


def compute_wasted_lane_fraction(n_elements, block_size):
    num_blocks = math.ceil(n_elements / block_size)
    total_lanes = num_blocks * block_size
    if total_lanes == 0:
        return 0.0
    wasted = total_lanes - n_elements
    return float(wasted / total_lanes)
