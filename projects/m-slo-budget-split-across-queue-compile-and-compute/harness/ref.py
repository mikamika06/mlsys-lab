def compute_breakdown(slo_ms, queue_depth, compile_overhead_ms, compute_per_token_ms, tokens):
    q_time = queue_depth * 0.5
    c_time = compile_overhead_ms
    comp_time = compute_per_token_ms * tokens
    total = q_time + c_time + comp_time
    return {"queue": q_time, "compile": c_time, "compute": comp_time, "total": total}


def find_optimal_batch(slo_ms, base_compile_ms, per_token_ms, max_batch):
    best = 1
    for b in range(1, max_batch + 1):
        time_cost = base_compile_ms + per_token_ms * b
        if time_cost <= slo_ms:
            best = b
    return float(best)


CASES = [
    {"slo": 100.0, "queue_depth": 10, "compile": 20.0, "per_token": 2.0, "tokens": 15},
    {"slo": 200.0, "queue_depth": 20, "compile": 15.0, "per_token": 1.5, "tokens": 40},
    {"slo": 50.0, "queue_depth": 2, "compile": 10.0, "per_token": 3.0, "tokens": 5}
]
