def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import quant
    import ref
    model, x = ref.get_fixture()
    shapes = {k: v.shape for k, v in model.items()}
    sens = ref.measure_sensitivity(model, x, [8, 4, 2])
    budget = 3000
    rec_ref = ref.build_recipe(shapes, sens, budget, [8, 4, 2])
    rec_stu = quant.build_recipe(shapes, sens, budget, [8, 4, 2])
    match = 1.0 if rec_ref == rec_stu else 0.0
    return {"recipe_match": match}
