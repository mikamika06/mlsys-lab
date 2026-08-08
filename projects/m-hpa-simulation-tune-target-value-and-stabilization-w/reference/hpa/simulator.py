import math

def simulate_hpa(metrics, initial_replicas, target_value, stabilization_window):
    replicas = []
    current = initial_replicas
    raw_desired = []
    for m in metrics:
        replicas.append(current)
        raw = max(1, math.ceil(current * m / target_value))
        raw_desired.append(raw)
        start = max(0, len(raw_desired) - stabilization_window)
        current = max(raw_desired[start:])
    return replicas

def tune_hpa(metrics, initial_replicas, configs):
    best_idx = -1
    best_cost = float('inf')
    for i, (target, window) in enumerate(configs):
        reps = simulate_hpa(metrics, initial_replicas, target, window)
        churn = sum(abs(reps[j] - reps[j-1]) for j in range(1, len(reps)))
        deficit = sum(max(0.0, metrics[j] - target * reps[j]) for j in range(len(metrics)))
        cost = churn * 100.0 + deficit
        if cost < best_cost:
            best_cost = cost
            best_idx = i
    return best_idx
