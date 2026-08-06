import math

CORES_TO_TEST = [1, 2, 4, 8, 16, 32, 64]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


def derive_config(hint: str, cores: int) -> tuple[int, int]:
    if hint == "latency":
        streams = 1
    elif hint == "throughput":
        streams = max(1, cores // 4)
    else:
        raise ValueError()

    threads = cores // streams
    return streams, threads


def estimate_throughput(batch_sizes: list[int], hint: str, cores: int) -> dict[int, float]:
    streams, threads = derive_config(hint, cores)
    res = {}

    for bs in batch_sizes:
        k = math.ceil(bs / streams)
        time_s = 2.0 + 0.5 * k + 0.1 * streams
        res[bs] = bs / time_s

    return res
