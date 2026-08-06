import ref


def check(workdir):
    try:
        from sens import metric, recipe
    except ImportError:
        return {"sens_matched": 0.0, "recipe_matched": 0.0}

    out = {"sens_matched": 0.0, "recipe_matched": 0.0}

    sens_ok = 0
    for layers in ref.CONFIGS:
        for layer in layers:
            want = ref.compute_true_sensitivity(layer)
            try:
                got = metric.compute_true_sensitivity(layer)
                rel_err = abs(want - got) / (abs(want) + 1e-9)
                if rel_err < 1e-3:
                    sens_ok += 1
            except Exception:
                pass

    out["sens_matched"] = 1.0 if sens_ok == 12 else 0.0

    recipe_ok = 0
    for layers in ref.CONFIGS:
        want_r = ref.build_recipe(layers)
        try:
            got_r = recipe.build_recipe(layers)
            match = True
            if not isinstance(got_r, list) or len(want_r) != len(got_r):
                match = False
            else:
                for wr, gr in zip(want_r, got_r):
                    if wr["layer_id"] != gr["layer_id"] or wr["bits"] != gr["bits"]:
                        match = False
                    rel_err = abs(wr["sensitivity"] - gr["sensitivity"]) / (abs(wr["sensitivity"]) + 1e-9)
                    if rel_err > 1e-3:
                        match = False
            if match:
                recipe_ok += 1
        except Exception:
            pass

    out["recipe_matched"] = 1.0 if recipe_ok == 3 else 0.0
    return out
