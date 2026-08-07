def check(workdir):
    import ref
    m = {"swap_cost_ok": 0.0}
    try:
        val = ref.swap_cost(1000, 65536, 32.0)
        if isinstance(val, (int, float)) and val > 0:
            m["swap_cost_ok"] = 1.0
    except Exception:
        pass
    return m
