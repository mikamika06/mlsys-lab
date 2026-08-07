def check(workdir):
    import numpy as np
    from gguf_recipe.evaluator import evaluate_recipe
    m = {"metrics_computed": 0.0}
    weights = {"layer1": np.random.randn(16, 16)}
    recipe = {"layer1": "Q4_0"}
    res = evaluate_recipe(weights, recipe)
    if isinstance(res, dict) and "ppl" in res and "kld" in res:
        m["metrics_computed"] = 1.0
    return m
