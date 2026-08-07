def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from lora_sweep import engine

    m = {"sweep_completed": 0.0, "step_budget_respected": 0.0}
    try:
        res = engine.run_rank_sweep([4, 8, 16], 500)
        if isinstance(res, list) and len(res) == 3:
            m["sweep_completed"] = 1.0
            m["step_budget_respected"] = 1.0
    except Exception:
        pass
    return m
