def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant.sensitivity import measure_sensitivity
    from quant.recipe import build_recipe
    from harness.ref import get_dummy_data

    model, dl = get_dummy_data()
    sens = measure_sensitivity(model, dl)
    m = {"budget_respected": 0.0}
    try:
        recipe = build_recipe(sens, budget_bits=4, allowed_bits=[2, 4])
        if isinstance(recipe, dict) and all(b in [2, 4] for b in recipe.values()):
            m["budget_respected"] = 1.0
    except Exception:
        pass
    return m
