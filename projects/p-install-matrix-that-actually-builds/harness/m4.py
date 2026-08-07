def check(workdir):
    from install.builder import build_on_config
    m = {"configs_passed": 0.0}
    configs = ["sm_80", "sm_89", "sm_90"]
    passed = 0
    for cfg in configs:
        try:
            res = build_on_config(cfg)
            if isinstance(res, dict) and res.get("status") == "success":
                passed += 1
        except Exception:
            pass
    m["configs_passed"] = float(passed)
    return m
