def check(workdir):
    from audit.core import analyze_fusion_gap
    m = {"gap_explained": 0.0}
    try:
        res = analyze_fusion_gap("test", "small")
        if isinstance(res, dict) and res.get("fused") is False:
            m["gap_explained"] = 1.0
    except Exception:
        pass
    return m
