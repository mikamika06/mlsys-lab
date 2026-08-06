CONFIGS = [
    """
# HELP vllm:gpu_cache_config_prefix_cache_hit_total Prefix cache hits
vllm:gpu_cache_config_prefix_cache_hit_total 10.0
vllm:prompt_tokens_total 100.0
""",
    """
# HELP vllm:gpu_cache_config_prefix_cache_hit_total Prefix cache hits
vllm:gpu_cache_config_prefix_cache_hit_total 55.0
vllm:prompt_tokens_total 200.0
""",
    """
# HELP vllm:gpu_cache_config_prefix_cache_hit_total Prefix cache hits
vllm:gpu_cache_config_prefix_cache_hit_total 120.0
vllm:prompt_tokens_total 500.0
"""
]

def parse_metrics(text):
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                metrics[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return metrics
