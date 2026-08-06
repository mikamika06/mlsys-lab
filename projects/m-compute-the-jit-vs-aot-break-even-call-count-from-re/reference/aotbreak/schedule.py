def select_strategy(profiles, expected_calls):
    """Select compilation mode and estimate latency for scheduled execution volumes."""
    out = {}
    for w, p in profiles.items():
        n = expected_calls.get(w, 1)
        jit_tot = p["jit_compile_ms"] + n * p["jit_exec_ms"]
        aot_tot = p["aot_load_ms"] + n * p["aot_exec_ms"]
        if jit_tot <= aot_tot:
            mode = "jit"
            est = jit_tot
        else:
            mode = "aot"
            est = aot_tot
        out[w] = {
            "selected_mode": mode,
            "estimated_latency_ms": float(est),
            "savings_ms": float(abs(jit_tot - aot_tot)),
        }
    return out
