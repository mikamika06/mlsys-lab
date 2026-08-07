def check(workdir):
    import ref
    from gbreak.optimizer import optimize_model, check_equivalence
    m = {"equivalent": 0.0}
    try:
        model = ref.get_sample_model()
        opt = optimize_model(model)
        if check_equivalence(model, opt, ref.get_sample_inputs()):
            m["equivalent"] = 1.0
    except Exception:
        pass
    return m
