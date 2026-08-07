def check(workdir):
    import numpy as np
    from gguf_recipe.quantizer import build_recipes
    m = {"recipes_generated": 0.0}
    weights = {"layer1": np.random.randn(16, 16)}
    imatrix = {"layer1": np.ones((16, 16))}
    r_with, r_without = build_recipes(weights, imatrix)
    if isinstance(r_with, dict) and isinstance(r_without, dict):
        if len(r_with) == 1 and len(r_without) == 1:
            m["recipes_generated"] = 1.0
    return m
