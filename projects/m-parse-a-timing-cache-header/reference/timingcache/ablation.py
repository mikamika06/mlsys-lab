def analyze_ablation(entries: list, allowed_tactic_sources: int) -> dict:
    if not entries:
        return {"retained_count": 0, "dropped_count": 0, "mean_latency_us": 0.0, "speedup": 1.0}

    retained = [e for e in entries if (e["tactic_source"] & allowed_tactic_sources) != 0]
    dropped = [e for e in entries if (e["tactic_source"] & allowed_tactic_sources) == 0]

    all_latencies = [e["latency_us"] for e in entries]
    retained_latencies = [e["latency_us"] for e in retained] if retained else [0.0]

    baseline_mean = sum(all_latencies) / len(all_latencies)
    retained_mean = sum(retained_latencies) / len(retained_latencies) if retained else 0.0

    speedup = baseline_mean / retained_mean if retained_mean > 0 else 0.0

    return {
        "retained_count": len(retained),
        "dropped_count": len(dropped),
        "mean_latency_us": float(retained_mean),
        "speedup": float(speedup),
    }
