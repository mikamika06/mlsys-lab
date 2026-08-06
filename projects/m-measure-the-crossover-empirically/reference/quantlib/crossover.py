def measure_scheme(scheme, workload):
    base_cost = scheme.get("bits", 4) * 10
    overhead = scheme.get("overhead", 5)
    intensity = workload.get("intensity", 1.0)
    simulated_throughput = float(workload.get("base_ops", 1000)) / (base_cost + overhead / intensity)
    return {"throughput": simulated_throughput, "scheme": scheme.get("name", "unknown")}

def find_crossover(schemes, workloads):
    best_crossover = None
    min_diff = float("inf")
    for w in workloads:
        results = [measure_scheme(s, w) for s in schemes]
        if len(results) < 2:
            continue
        diff = abs(results[0]["throughput"] - results[1]["throughput"])
        if diff < min_diff:
            min_diff = diff
            best_crossover = {"workload_id": w.get("id"), "throughput_ratio": results[0]["throughput"] / results[1]["throughput"]}
    return best_crossover
