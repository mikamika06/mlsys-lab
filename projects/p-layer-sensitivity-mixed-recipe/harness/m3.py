def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import quant
    import ref
    model, x = ref.get_fixture()
    recipe = {"l1": 8, "l2": 4, "l3": 2, "l4": 8}
    val_ref = ref.evaluate_recipe(model, x, recipe)
    val_stu = quant.evaluate_recipe(model, x, recipe)
    match = 1.0 if abs(val_ref - val_stu) < 1e-4 else 0.0
    return {"eval_match": match}
