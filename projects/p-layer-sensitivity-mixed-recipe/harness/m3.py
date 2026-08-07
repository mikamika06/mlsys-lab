def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant.sensitivity import measure_sensitivity
    from quant.recipe import build_recipe
    from quant.mixed import apply_mixed_quantization, evaluate_model
    from harness.ref import get_dummy_data

    model, dl = get_dummy_data()
    sens = measure_sensitivity(model, dl)
    recipe = build_recipe(sens, budget_bits=4, allowed_bits=[2, 4])
    m = {"end_to_end_ok": 0.0}
    try:
        q_model = apply_mixed_quantization(model, recipe)
        loss = evaluate_model(q_model, dl)
        if isinstance(loss, float) and loss >= 0.0:
            m["end_to_end_ok"] = 1.0
    except Exception:
        pass
    return m
