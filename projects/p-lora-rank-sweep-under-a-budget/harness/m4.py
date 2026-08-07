def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from lora_sweep import optimizer

    m = {"scaling_analyzed": 0.0}
    try:
        res = optimizer.analyze_alpha_scaling([16, 32], [8, 16])
        if isinstance(res, dict) and "scaling_analyzed" in res:
            m["scaling_analyzed"] = float(res["scaling_analyzed"])
    except Exception:
        pass
    return m
