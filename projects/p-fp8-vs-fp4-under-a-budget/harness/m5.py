def check(workdir):
    import sys
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from quant.selector import recommend_format

    m = {"recommendation_ok": 0.0}
    candidates = [("fp8", 8.25, 0.01), ("fp4", 4.25, 0.08)]
    rec = recommend_format(candidates, 5.0, "blackwell")
    if rec == "fp4":
        m["recommendation_ok"] = 1.0
    return m
