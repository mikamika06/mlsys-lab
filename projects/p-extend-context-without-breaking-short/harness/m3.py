def check(workdir):
    from ctx.scaling import tune_parameters
    import ref
    m = {"dual_params_tuned": 0.0}
    try:
        res = tune_parameters({"default_param": 4.0}, ref.get_short_inputs(), ref.get_long_inputs())
        if isinstance(res, dict) and "scale_factor" in res:
            m["dual_params_tuned"] = 1.0
    except Exception:
        pass
    return m
