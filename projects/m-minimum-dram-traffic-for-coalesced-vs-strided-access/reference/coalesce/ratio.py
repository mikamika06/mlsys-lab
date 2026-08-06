from coalesce.traffic import min_dram_traffic
from coalesce.simulate import simulate_warp_coalescing


def excess_traffic_ratio(num_elements, element_size, stride):
    actual_traffic = min_dram_traffic(num_elements, element_size, stride)
    ideal_traffic = min_dram_traffic(num_elements, element_size, 1)
    if ideal_traffic == 0:
        return 1.0
    return float(actual_traffic) / float(ideal_traffic)
