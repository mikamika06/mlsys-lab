import ref

def check(workdir):
    from router.sim import simulate_trace
    trace = ref.get_reference_trace()
    got = simulate_trace(3, trace, "cache_aware_guardrail")
    want = ref.simulate_trace(3, trace, "cache_aware_guardrail")
    out = {"comparison_match": 1.0 if abs(got - want) < 1e-5 else 0.0}
    if abs(got - want) >= 1e-5:
        out["_note"] = f"got hit rate {got}, want {want}"
    return out
