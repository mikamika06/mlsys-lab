def compare_prompts(single_metrics: dict, multi_metrics: dict) -> dict:
    return {
        "latency_ratio": multi_metrics["latency"] / max(single_metrics["latency"], 1e-5),
        "memory_ratio": multi_metrics["memory"] / max(single_metrics["memory"], 1e-5),
        "valid": True
    }
