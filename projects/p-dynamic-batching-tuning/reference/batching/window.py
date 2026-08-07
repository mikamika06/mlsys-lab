def find_optimal_window(latency_curve, target_slo):
    best_w = 0.0
    best_throughput = 0.0
    for b, lat in latency_curve.items():
        if lat <= target_slo:
            throughput = b / (lat / 1000.0)
            if throughput > best_throughput:
                best_throughput = throughput
                best_w = float(lat * 0.5)
    return {"optimal_window": best_w, "max_throughput": best_throughput}
