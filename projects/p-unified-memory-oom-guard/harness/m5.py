def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from guard.limiter import run_safe_mode

    m = {"survived_large": 0.0}
    configs = ref.get_large_configs()
    try:
        res = run_safe_mode(configs)
        if isinstance(res, list) and len(res) == len(configs):
            m["survived_large"] = 1.0
    except Exception:
        pass
    return m
