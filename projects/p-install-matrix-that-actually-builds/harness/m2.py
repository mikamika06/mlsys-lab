def check(workdir):
    from install.matrix import analyze_gates
    m = {"bottlenecks_identified": 0.0, "exclude_invalid": 0.0}
    try:
        res = analyze_gates()
    except Exception:
        return m
    if isinstance(res, dict) and "blocked_by" in res:
        m["bottlenecks_identified"] = 1.0
        m["exclude_invalid"] = 1.0 if len(res["blocked_by"]) > 0 else 0.0
    return m
