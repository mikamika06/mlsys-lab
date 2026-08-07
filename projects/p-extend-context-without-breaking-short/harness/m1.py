def check(workdir):
    from ctx.scaling import measure_short_degradation
    import ref
    m = {"short_degradation_measured": 0.0}
    try:
        model = lambda x: float(x)
        res = measure_short_degradation(model, ref.get_short_inputs())
        if isinstance(res, float):
            m["short_degradation_measured"] = 1.0
    except Exception:
        pass
    return m
