import ref


def check(workdir):
    from gguf_quant.analyzer import quantify_requantize_error, compare_recipes

    weights = ref.get_sample_weights()
    err = quantify_requantize_error(weights, 0.5)
    res = compare_recipes(weights)

    out = {"error_quantified": 0.0, "recipe_compared": 0.0}
    if isinstance(err, float):
        out["error_quantified"] = 1.0
    if isinstance(res, dict) and "default" in res and "pure" in res:
        out["recipe_compared"] = 1.0
    return out
