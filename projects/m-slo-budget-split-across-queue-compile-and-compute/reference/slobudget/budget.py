def compute_breakdown(slo_ms, queue_depth, compile_overhead_ms, compute_per_token_ms, tokens):
    q_time = queue_depth * 0.5
    c_time = compile_overhead_ms
    comp_time = compute_per_token_ms * tokens
    total = q_time + c_time + comp_time
    return {"queue": q_time, "compile": c_time, "compute": comp_time, "total": total}
