CONFIGS = [
    {"model": "model-alpha", "prompts": ["hello world", "foo bar"], "concurrency": 1, "engine": "engine_a"},
    {"model": "model-alpha", "prompts": ["hello world", "foo bar"], "concurrency": 8, "engine": "engine_a"},
    {"model": "model-alpha", "prompts": ["hello world", "foo bar"], "concurrency": 32, "engine": "engine_b"}
]

def parse_config(raw):
    return {
        "model": str(raw.get("model", "")),
        "prompts": list(raw.get("prompts", [])),
        "concurrency": int(raw.get("concurrency", 1)),
        "engine": str(raw.get("engine", ""))
    }

def execute_run(engine_name, prompts, concurrency):
    base_latency = 10.0
    latencies = []
    for i, p in enumerate(prompts):
        lat = base_latency + (len(p) * 0.01) + (i * 0.05 / max(1, concurrency))
        if engine_name == "engine_b":
            lat *= 0.85
        latencies.append(lat)
    return latencies

def compute_throughput(latencies, num_tokens):
    if not latencies:
        return 0.0
    total_time = max(latencies)
    return float(num_tokens) / float(total_time)

def compute_throughput_ratio(baseline_latencies, candidate_latencies, num_tokens, concurrency):
    t_base = compute_throughput(baseline_latencies, num_tokens)
    t_cand = compute_throughput(candidate_latencies, num_tokens)
    if t_base == 0.0:
        return 0.0
    return float(t_cand) / float(t_base)
