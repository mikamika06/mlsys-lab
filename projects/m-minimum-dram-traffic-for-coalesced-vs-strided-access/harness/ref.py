def min_dram_traffic(num_elements, element_size, stride):
    if num_elements <= 0:
        return 0
    sectors = set()
    for i in range(num_elements):
        addr = i * stride * element_size
        start_sector = addr // 32
        end_sector = (addr + element_size - 1) // 32
        for s in range(start_sector, end_sector + 1):
            sectors.add(s)
    return len(sectors) * 32


def simulate_warp_coalescing(addresses, element_size=4):
    sectors = set()
    for addr in addresses:
        start_sector = addr // 32
        end_sector = (addr + element_size - 1) // 32
        for s in range(start_sector, end_sector + 1):
            sectors.add(s)
    return len(sectors)


def excess_traffic_ratio(num_elements, element_size, stride):
    actual_traffic = min_dram_traffic(num_elements, element_size, stride)
    ideal_traffic = min_dram_traffic(num_elements, element_size, 1)
    if ideal_traffic == 0:
        return 1.0
    return float(actual_traffic) / float(ideal_traffic)


TEST_CASES = [
    {"num_elements": 32, "element_size": 4, "stride": 1},
    {"num_elements": 32, "element_size": 4, "stride": 2},
    {"num_elements": 32, "element_size": 4, "stride": 4},
]


WARP_TEST_CASES = [
    [i * 4 for i in range(32)],
    [i * 8 for i in range(32)],
    [i * 16 for i in range(32)],
]
