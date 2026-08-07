def check(workdir):
    m = {"invariant_holds": 0.0}
    try:
        from system import analysis
        diff = analysis.simulate_data_invariant(1000, 64)
        if diff == 0.0:
            m["invariant_holds"] = 1.0
    except Exception:
        pass
    return m
