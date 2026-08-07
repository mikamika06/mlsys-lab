import math

def ring_time(size_bytes, num_ranks, alpha, beta):
    return 2 * (num_ranks - 1) / num_ranks * alpha + (num_ranks - 1) / num_ranks * size_bytes * beta

def tree_time(size_bytes, num_ranks, alpha, beta):
    steps = math.ceil(math.log2(num_ranks))
    return 2 * steps * alpha + 2 * (num_ranks - 1) / num_ranks * size_bytes * beta

def find_crossover(alpha, beta, num_ranks):
    low = 0.0
    high = 1e9
    for _ in range(100):
        mid = (low + high) / 2
        if ring_time(mid, num_ranks, alpha, beta) < tree_time(mid, num_ranks, alpha, beta):
            low = mid
        else:
            high = mid
    return low
