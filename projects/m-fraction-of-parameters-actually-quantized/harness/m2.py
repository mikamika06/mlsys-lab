import ref


def check(workdir):
    from quant_target import metrics, targeting
    out = {"fraction_match": 0.0, "head_cost_match": 0.0}
    ok_frac = 0
    ok_cost = 0
    total = len(ref.CONFIGS)
    for cfg in ref.CONFIGS:
        targets = targeting.filter_target_modules(cfg)
        want_frac, want_cost = ref.compute_quantized_fraction(cfg, targets)
        got_frac, got_cost = metrics.compute_quantized_fraction(cfg, targets)
        if abs(got_frac - want_frac) < 1e-5:
            ok_frac += 1
        if got_cost == want_cost:
            ok_cost += 1
    out["fraction_match"] = 1.0 if ok_frac == total else 0.0
    out["head_cost_match"] = 1.0 if ok_cost == total else 0.0
    return out
