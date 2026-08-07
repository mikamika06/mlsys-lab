import ref


def check(workdir):
    from specalpha.metrics import expected_speedup
    from specalpha.reconstruct import reconstruct_alphas
    out = {"metrics_match": 0.0, "bound_check": 0.0}
    ok_metrics = 0
    ok_bounds = 0
    for profile in ref.PROFILES:
        alphas = reconstruct_alphas(profile["histogram"], profile["max_k"])
        got_speedup = expected_speedup(alphas)
        want_speedup = ref.generate_reference_metrics(profile)
        if abs(got_speedup - want_speedup) < 1e-5:
            ok_metrics += 1
        if all(0.0 <= a <= 1.0 for a in alphas):
            ok_bounds += 1
    if ok_metrics == len(ref.PROFILES):
        out["metrics_match"] = 1.0
    if ok_bounds == len(ref.PROFILES):
        out["bound_check"] = 1.0
    return out
