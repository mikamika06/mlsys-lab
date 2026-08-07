def check(workdir):
    from ctx.scaling import evaluate_dual_regime
    import ref
    m = {"short_no_regression": 0.0, "long_improved": 0.0}
    try:
        model = lambda x: 0.96
        res = evaluate_dual_regime(model, ref.get_short_inputs(), ref.get_long_inputs(), ref.get_baseline_score())
        if res.get("short_ok") == 1.0 and res.get("long_ok") == 1.0:
            m["short_no_regression"] = 1.0
            m["long_improved"] = 1.0
    except Exception:
        pass
    return m
