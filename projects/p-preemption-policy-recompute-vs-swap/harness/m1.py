def check(workdir):
    import ref
    m = {"recompute_cost_ok": 0.0}
    try:
        val = ref.recompute_cost(1000, 4096, 32, 300.0)
        if isinstance(val, (int, float)) and val > 0:
            m["recompute_cost_ok"] = 1.0
    except Exception:
        pass
    return m
