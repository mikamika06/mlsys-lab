def compute_blocking(latencies, prompt_tokens):
    if not latencies:
        return 0.0
    baseline = min(latencies)
    peak = max(latencies)
    ratio = peak / (baseline if baseline > 0 else 1.0)
    scale_factor = prompt_tokens / 1024.0
    return float(peak - baseline + scale_factor * ratio)
