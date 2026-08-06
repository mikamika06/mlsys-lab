def parse_config(raw_config):
    if not isinstance(raw_config, dict):
        raise ValueError("config must be a dictionary")
    return {
        "model": str(raw_config.get("model", "")),
        "prompts": list(raw_config.get("prompts", [])),
        "concurrency": int(raw_config.get("concurrency", 1)),
        "engine": str(raw_config.get("engine", ""))
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
