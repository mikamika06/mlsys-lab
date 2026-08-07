def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from lora_sweep import config, engine

    m = {"baseline_loss": 99.0, "eval_stable": 0.0}
    try:
        cfg = config.get_default_config()
        res = engine.run_baseline(cfg)
        m["baseline_loss"] = float(res.get("loss", 99.0))
        m["eval_stable"] = float(res.get("eval_stable", 0.0))
    except Exception:
        pass
    return m
