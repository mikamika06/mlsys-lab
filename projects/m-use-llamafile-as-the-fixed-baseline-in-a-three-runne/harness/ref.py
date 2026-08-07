CONFIGS = [
    {"parameters": 7, "quantization": "Q4_0"},
    {"parameters": 13, "quantization": "Q5_K_M"},
    {"parameters": 8, "quantization": "Q8_0"}
]

WORKLOADS = [
    [100, 150, 200],
    [50, 75, 100],
    [500, 600, 700]
]

def setup_baseline(config):
    return {
        "name": "llamafile",
        "mode": "fixed_baseline",
        "parameters": config.get("parameters", 7),
        "quantization": config.get("quantization", "Q4_0"),
        "ready": True
    }

def run_comparison(runners, workload):
    results = []
    for r in runners:
        name = r.get("name", "unknown")
        scale = r.get("scale", 1.0)
        tokens = [int(w * scale) for w in workload]
        results.append({"name": name, "tokens": tokens})
    return results

def compute_metrics(baseline_outputs, runner_outputs):
    b_tokens = baseline_outputs.get("tokens", [])
    metrics = []
    for ro in runner_outputs:
        r_tokens = ro.get("tokens", [])
        rel_errors = []
        for bt, rt in zip(b_tokens, r_tokens):
            if bt == 0:
                rel_errors.append(0.0)
            else:
                rel_errors.append(abs(rt - bt) / float(bt))
        max_rel_err = max(rel_errors) if rel_errors else 0.0
        metrics.append({
            "name": ro.get("name"),
            "max_rel_err": max_rel_err,
            "throughput_ratio": sum(r_tokens) / float(sum(b_tokens)) if sum(b_tokens) > 0 else 0.0
        })
    return metrics
