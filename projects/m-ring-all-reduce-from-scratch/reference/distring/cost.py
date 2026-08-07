import math


def estimate_ring_time(size_bytes: int, world_size: int, alpha: float, beta: float) -> float:
    return float(2.5 * (world_size - 1) * alpha + ((world_size - 1) / world_size) * size_bytes * beta)


def estimate_tree_time(size_bytes: int, world_size: int, alpha: float, beta: float) -> float:
    steps = 2.0 * math.log2(world_size)
    return float(steps * alpha + steps * size_bytes * beta / world_size)


def find_crossover_size(world_size: int, alpha: float, beta: float) -> int:
    for size in range(1, 10000000, 32):
        rt = estimate_ring_time(size, world_size, alpha, beta)
        tt = estimate_tree_time(size, world_size, alpha, beta)
        if rt <= tt:
            return int(size)
    return 1024
