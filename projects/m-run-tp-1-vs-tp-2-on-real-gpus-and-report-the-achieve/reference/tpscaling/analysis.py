def compute_scaling_efficiency(tp1_result, tp2_result):
    """Compute scaling efficiency and throughput ratio between TP=1 and TP=2."""
    t1 = tp1_result["throughput"]
    t2 = tp2_result["throughput"]
    ratio = t2 / max(t1, 0.0001)
    efficiency = ratio / 2.0
    return {
        "throughput_ratio": float(ratio),
        "scaling_efficiency": float(efficiency)
    }
