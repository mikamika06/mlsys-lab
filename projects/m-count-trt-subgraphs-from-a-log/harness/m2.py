import ref


def check(workdir):
    from trtlog.bench import compare_eps, simulate_cache_sessions
    out = {"latency_ratio_match": 0.0, "cache_reuse_match": 0.0}
    c_lats = [15.0, 16.0, 14.0]
    t_lats = [5.0, 5.5, 4.5]
    want_comp = ref.compare_eps(c_lats, t_lats)
    got_comp = compare_eps(c_lats, t_lats)

    if abs(got_comp["latency_ratio"] - want_comp["latency_ratio"]) < 1e-5:
        out["latency_ratio_match"] = 1.0
    else:
        out["_note"] = f"latency ratio mismatch: got {got_comp}, want {want_comp}"

    want_cache = ref.simulate_cache_sessions(ref.LOGS)
    got_cache = simulate_cache_sessions(ref.LOGS)
    if got_cache == want_cache:
        out["cache_reuse_match"] = 1.0
    else:
        out["_note"] = f"cache reuse mismatch: got {got_cache}, want {want_cache}"
    return out
