import ref


def check(workdir):
    from trtopt.simulate import simulate_performance
    out = {"sim_match": 0.0}
    cases = [
        (256, "static", 10.0, [128, 256]),
        (1024, "continuous", 15.0, [256, 512, 1024])
    ]
    ok = 0
    for max_tok, btype, rate, prefill in cases:
        want_t, want_tp = ref.simulate_performance(max_tok, btype, rate, prefill)
        got_t, got_tp = simulate_performance(max_tok, btype, rate, prefill)
        if abs(want_t - got_t) < 1e-5 and abs(want_tp - got_tp) < 1e-5:
            ok += 1
    if ok == len(cases):
        out["sim_match"] = 1.0
    return out
