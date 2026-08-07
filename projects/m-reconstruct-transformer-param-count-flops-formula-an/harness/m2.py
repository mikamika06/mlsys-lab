import ref

def check(workdir):
    from studentdesign.init_loss import evaluate_init_strategies
    out = {"strategies_matched": 0.0}
    cfg = ref.CONFIGS[0]
    strategies = ["random", "stacked", "truncated"]
    ok = True
    for s in strategies:
        want = ref.measure_init_loss(cfg, s)
        got = evaluate_init_strategies(cfg, s)
        if abs(got - want) > 1e-5:
            ok = False
    out["strategies_matched"] = 1.0 if ok else 0.0
    return out
