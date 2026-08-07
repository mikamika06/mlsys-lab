import numpy as np


def setup_stand(config):
    return {
        "context_len": config.get("context_len", 512),
        "batch_size": config.get("batch_size", 1),
        "quantization": config.get("quantization", "4bit")
    }


def run_benchmark(engine, workload, runs=3):
    times_prefill = []
    times_decode = []
    for _ in range(runs):
        p = engine.run_prefill(workload["tokens"])
        d = engine.run_decode(workload["decode_steps"])
        times_prefill.append(p["time_ms"])
        times_decode.append(d["time_ms"])

    m = engine.get_metrics()
    return {
        "engine": engine.name,
        "prefill_mean": float(np.mean(times_prefill)),
        "prefill_std": float(np.std(times_prefill)),
        "decode_mean": float(np.mean(times_decode)),
        "decode_std": float(np.std(times_decode)),
        "memory_mb": m["memory_mb"],
        "energy_j": m["energy_j"]
    }


def analyze_results(results):
    best = min(results, key=lambda x: x["decode_mean"])
    return {
        "recommended": best["engine"],
        "valid": True
    }
