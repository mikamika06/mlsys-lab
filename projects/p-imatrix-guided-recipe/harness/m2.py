def check(workdir):
    import numpy as np
    from gguf_recipe.selector import select_quant_types
    m = {"selection_shifted": 0.0}
    weights = {"layer1": np.random.randn(16, 16)}
    imatrix = {"layer1": np.ones((16, 16)) * 5.0}
    res = select_quant_types(weights, imatrix, threshold=1.0)
    if res.get("layer1") == "Q8_0":
        m["selection_shifted"] = 1.0
    return m
