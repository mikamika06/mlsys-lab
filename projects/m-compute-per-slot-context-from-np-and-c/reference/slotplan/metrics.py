def parse_metrics(metrics_text):
    """Parse llama-server /metrics endpoint output."""
    metrics = {}
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2:
            key, val = parts[0], parts[1]
            try:
                metrics[key] = float(val)
            except ValueError:
                pass
    return metrics


def compute_cache_reuse_ratio(metrics_text):
    """Compute prompt cache reuse ratio from parsed metrics."""
    metrics = parse_metrics(metrics_text)
    hits = metrics.get("llamacpp:prompt_tokens_seconds_sum", 0.0)
    eval_tokens = metrics.get("llamacpp:prompt_tokens_processed_total", 0.0)
    cached_tokens = metrics.get("llamacpp:prompt_tokens_cached_total", 0.0)

    total = eval_tokens + cached_tokens
    if total <= 0:
        return 0.0
    return cached_tokens / total
