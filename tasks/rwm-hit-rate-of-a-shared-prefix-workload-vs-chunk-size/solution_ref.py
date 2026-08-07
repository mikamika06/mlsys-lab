import math


def prefix_chunk_hit_rate(prefix_lengths: list[int], chunk_size: int) -> float:
    c = float(chunk_size)
    total = 0.0
    n = len(prefix_lengths)
    for i in range(n):
        val = float(prefix_lengths[i])
        reused = math.floor(val / c) * c
        total += reused / val
    return float(total / n)
