import ref

def check(workdir):
    out = {"p99_reduced": 0.0}
    try:
        from policy.policy import PreemptionPolicy
        p = PreemptionPolicy(1000.0, 100000.0, 250.0, 2.0, 0.1)
        trace = ref.generate_trace(1000, 42)

        smart_lats = p.evaluate_trace(trace, "smart")
        recomp_lats = p.evaluate_trace(trace, "recompute")
        swap_lats = p.evaluate_trace(trace, "swap")

        smart_p99 = ref.p99(smart_lats)
        recomp_p99 = ref.p99(recomp_lats)
        swap_p99 = ref.p99(swap_lats)

        if smart_p99 <= recomp_p99 and smart_p99 <= swap_p99 and smart_p99 < max(recomp_p99, swap_p99):
            out["p99_reduced"] = 1.0

        return out
    except Exception:
        return out
