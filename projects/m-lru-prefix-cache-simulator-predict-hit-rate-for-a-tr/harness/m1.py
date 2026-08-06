import ref

def check(workdir):
    from prefcache.simulator import simulate_lru_hit_rate
    out = {"hit_rate_matched": 0.0}
    ok = 0
    for traces, cap in ref.TRACES_SET:
        want = ref.simulate_lru_hit_rate(traces, cap)
        try:
            got = simulate_lru_hit_rate(traces, cap)
        except Exception:
            got = -1.0
        if isinstance(got, (int, float)) and abs(got - want) < 1e-5:
            ok += 1
    out["hit_rate_matched"] = float(ok)
    return out
