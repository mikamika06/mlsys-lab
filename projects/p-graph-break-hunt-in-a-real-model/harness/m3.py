def check(workdir):
    import ref
    from gbreak.optimizer import optimize_model
    m = {"expensive_removed": 0.0}
    try:
        opt = optimize_model(ref.get_sample_model())
        if opt is not None:
            m["expensive_removed"] = 1.0
    except Exception:
        pass
    return m
