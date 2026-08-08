def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import quant
    import ref
    model, x = ref.get_fixture()
    budget = 3000
    out_ref = ref.compare_recipes(model, x, budget, [8, 4, 2])
    out_stu = quant.compare_recipes(model, x, budget, [8, 4, 2])
    match = 1.0
    for k in out_ref:
        if isinstance(out_ref[k], dict):
            if out_ref[k] != out_stu.get(k): match = 0.0
        else:
            if abs(out_ref[k] - out_stu.get(k, 0.0)) > 1e-4: match = 0.0
    return {"compare_match": match}
