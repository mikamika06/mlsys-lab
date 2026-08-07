import numpy as np


def measure_ttft_gain(requests, config):
    ratios = []
    for req in requests:
        prompt_len = req["prompt_len"]
        hit_rate = req["cache_hit_rate"]
        chunk_size = config.get("chunk_size", 256)
        base_time = prompt_len * 0.05
        transfer_overhead = (
            (prompt_len * hit_rate) / chunk_size
        ) * 0.002 + 0.005 if hit_rate > 0 else 0
        offload_time = (prompt_len * (1 - hit_rate) * 0.05) + transfer_overhead
        ratio = offload_time / base_time if base_time > 0 else 1.0
        ratios.append(float(ratio))
    return {"latency_ratios": ratios, "mean_ratio": float(np.mean(ratios))}
