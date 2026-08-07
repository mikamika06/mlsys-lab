import ref


def check(workdir):
    from kvquant.recipe import make_recipe
    from kvquant.serving import build_serving_args

    out = {"serving_configured": 0.0}
    cfg = ref.CONFIGS[0]
    recipe = make_recipe(cfg, scheme="fp8", block_size=16)
    got_args = build_serving_args(recipe, model_path="/models/test")
    want_args = ref.build_serving_args(recipe, model_path="/models/test")
    if got_args == want_args:
        out["serving_configured"] = 1.0
    else:
        out["_note"] = f"got args {got_args}, want {want_args}"
    return out
