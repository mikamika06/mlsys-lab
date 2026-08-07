def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from lora_sweep import optimizer

    m = {"modules_evaluated": 0.0}
    try:
        res = optimizer.evaluate_modules([["q_proj"], ["v_proj"], ["q_proj", "v_proj"]])
        if isinstance(res, list) and len(res) >= 3:
            m["modules_evaluated"] = float(len(res))
    except Exception:
        pass
    return m
