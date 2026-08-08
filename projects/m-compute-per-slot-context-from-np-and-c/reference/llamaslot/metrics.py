def parse_prometheus_metrics(raw_text: str) -> dict:
    """Parse Prometheus text format into a metric key-value dictionary."""
    out = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        metric_expr, val_str = parts[0], parts[1]
        metric_name = metric_expr.split("{")[0]
        try:
            val = float(val_str)
            out[metric_name] = out.get(metric_name, 0.0) + val
        except ValueError:
            continue
    return out


def analyze_prompt_cache(raw_text: str) -> dict:
    """Analyze prompt cache metrics to evaluate total, processed, and cached tokens."""
    metrics = parse_prometheus_metrics(raw_text)
    total_prompt = metrics.get("llamacpp:prompt_tokens_total", 0.0)
    eval_prompt = metrics.get("llamacpp:prompt_tokens_processed", 0.0)
    if "llamacpp:prompt_tokens_cached" in metrics:
        cached_prompt = metrics["llamacpp:prompt_tokens_cached"]
    else:
        cached_prompt = max(0.0, total_prompt - eval_prompt)
    hit_ratio = (cached_prompt / total_prompt) if total_prompt > 0 else 0.0
    return {
        "prompt_tokens_total": int(total_prompt),
        "prompt_tokens_processed": int(eval_prompt),
        "prompt_tokens_cached": int(cached_prompt),
        "hit_ratio": float(hit_ratio),
        "is_reusing": hit_ratio > 0.0,
    }
