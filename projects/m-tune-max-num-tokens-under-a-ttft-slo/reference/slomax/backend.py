def estimate_step_latency(num_tokens: int, num_seqs: int, backend_type: str, batching_mode: str) -> float:
    if backend_type == "pytorch":
        base = 0.8
        per_tok = 0.005
        per_seq = 0.05
    elif backend_type == "cpp":
        base = 0.2
        per_tok = 0.004
        per_seq = 0.02
    else:
        raise ValueError(f"Unknown backend_type: {backend_type}")

    lat = base + per_tok * num_tokens + per_seq * num_seqs
    if batching_mode == "static":
        lat *= 1.2
    elif batching_mode != "continuous":
        raise ValueError(f"Unknown batching_mode: {batching_mode}")
    return lat


def compute_ttft(prompt_length: int, max_num_tokens: int, backend_type: str, batching_mode: str) -> float:
    if max_num_tokens <= 0 or prompt_length <= 0:
        return float("inf")

    if batching_mode == "static":
        if prompt_length > max_num_tokens:
            return float("inf")
        return estimate_step_latency(prompt_length, 1, backend_type, "static")
    elif batching_mode == "continuous":
        total_lat = 0.0
        remaining = prompt_length
        while remaining > 0:
            chunk = min(remaining, max_num_tokens)
            total_lat += estimate_step_latency(chunk, 1, backend_type, "continuous")
            remaining -= chunk
        return total_lat
    else:
        raise ValueError(f"Unknown batching_mode: {batching_mode}")
