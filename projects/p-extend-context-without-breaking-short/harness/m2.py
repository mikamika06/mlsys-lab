def check(workdir):
    from ctx.scaling import compare_scaling_methods
    import ref
    m = {"methods_compared": 0.0}
    try:
        res = compare_scaling_methods(ref.get_eval_methods(), ref.get_short_inputs())
        if isinstance(res, dict) and len(res) > 0:
            m["methods_compared"] = 1.0
    except Exception:
        pass
    return m
