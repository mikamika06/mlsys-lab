import math

def crossover_batch_size(peak_compute: float,
                         peak_mem: float,
                         compute_per_token: float,
                         mem_per_token: float) -> int:
    theta = peak_compute / peak_mem
    val = (theta * mem_per_token) / compute_per_token
    return math.ceil(val**2)
