def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant.sensitivity import measure_sensitivity
    from quant.recipe import build_recipe
    from quant.mixed import apply_mixed_quantization, evaluate_model
    from harness.ref import get_dummy_data

    model, dl = get_dummy_data()
    sens = measure_sensitivity(model, dl)
    recipe = build_recipe(sens, budget_bits=3, allowed_bits=[2, 4])

    uniform_recipe = {k: 3 for k in model.keys()}

    q_mixed = apply_mixed_quantization(model, recipe)
    q_uni = apply_mixed_quantization(model, uniform_recipe)

    loss_mixed = evaluate_model(q_mixed, dl)
    loss_uni = evaluate_model(q_uni, dl)

    m = {"beats_uniform": 0.0}
    if loss_mixed <= loss_uni:
        m["beats_uniform"] = 1.0
    return m
