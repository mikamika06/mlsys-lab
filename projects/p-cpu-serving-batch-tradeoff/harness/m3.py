def check(workdir):
    from serving import engine
    m = {"slo_point_ok": 0.0}
    try:
        bs_list = [1, 2, 4, 8, 16, 32]
        opt = engine.find_slo_point(bs_list, 50.0, 10.0, 4)
        if opt in bs_list:
            m["slo_point_ok"] = 1.0
    except Exception:
        pass
    return m
