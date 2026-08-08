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

def quantify_hit_rate_loss(session_turns, num_replicas):
    if not session_turns:
        return 0.0
    total = sum(session_turns)
    if total == 0:
        return 0.0
    possible = sum(t - 1 for t in session_turns if t > 1)
    perfect = possible / total
    rand = (possible / num_replicas) / total
    return perfect - rand

METRICS_1 = [50.0 + 40.0 * math.sin(i / 2.0) for i in range(50)]
CONFIGS_1 = [(10.0, 1), (10.0, 5), (20.0, 1), (20.0, 5), (30.0, 10)]

SESSIONS = [
    [1, 1, 1, 1],
    [5, 5, 5],
    [10, 2, 3, 1],
    [100, 50, 10]
]
