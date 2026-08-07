def infer_residency(ttft_samples: list, baseline_latency: float) -> bool:
    if not ttft_samples:
        return False
    avg_sample = sum(ttft_samples) / len(ttft_samples)
    return avg_sample < (baseline_latency * 0.7)
