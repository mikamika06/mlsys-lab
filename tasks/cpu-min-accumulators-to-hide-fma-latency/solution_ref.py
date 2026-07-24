from math import ceil


def min_fma_accumulators(latency: int, throughput: float, length: int, line_bytes: int, sets: int, ways: int):
    accumulators = ceil(latency * throughput)
    addresses = [i * line_bytes for i in range(length)]
    return accumulators, addresses
