CONFIGS = [
    {"steps": 5, "base_ttft": 10.0, "base_itl": 2.0},
    {"steps": 8, "base_ttft": 12.0, "base_itl": 2.5},
    {"steps": 10, "base_ttft": 8.0, "base_itl": 1.5},
]

def simulate_sweep(config):
    steps = config.get("steps", 10)
    base_ttft = config.get("base_ttft", 10.0)
    base_itl = config.get("base_itl", 2.0)
    results = []
    for i in range(steps):
        load_factor = 1.0 + (i * 0.2)
        ttft = base_ttft * load_factor
        itl = base_itl * (1.0 + (i * 0.15))
        results.append({"step": i, "ttft": ttft, "itl": itl})
    return results

def compute_blocking(latencies, prompt_tokens):
    if not latencies:
        return 0.0
    baseline = min(latencies)
    peak = max(latencies)
    ratio = peak / (baseline if baseline > 0 else 1.0)
    scale_factor = prompt_tokens / 1024.0
    return float(peak - baseline + scale_factor * ratio)

def align_chunks(chunk_size, block_size):
    remainder = chunk_size % block_size
    if remainder == 0:
        return chunk_size
    return chunk_size + (block_size - remainder)
