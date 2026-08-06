import math
import os


def derive_config(hint: str, cores: int) -> tuple[int, int]:
    if hint == "latency":
        streams = 1
    elif hint == "throughput":
        streams = max(1, cores // 4)
    else:
        raise ValueError("Invalid hint")

    threads = cores // streams
    return streams, threads


def compile_model(model_name: str, cache_dir: str) -> float:
    path = os.path.join(cache_dir, f"{model_name}.bin")
    if os.path.exists(path):
        return 0.1

    with open(path, "wb") as f:
        f.write(b"compiled")
    return 5.0


def estimate_throughput(batch_sizes: list[int], hint: str, cores: int) -> dict[int, float]:
    streams, threads = derive_config(hint, cores)
    results = {}

    for bs in batch_sizes:
        k = math.ceil(bs / streams)
        time_s = 2.0 + 0.5 * k + 0.1 * streams
        results[bs] = bs / time_s

    return results
