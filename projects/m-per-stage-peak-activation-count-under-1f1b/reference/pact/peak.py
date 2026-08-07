def compute_peak_activations(num_stages: int, num_microbatches: int) -> list[int]:
    return [min(num_stages - s, num_microbatches) for s in range(num_stages)]
