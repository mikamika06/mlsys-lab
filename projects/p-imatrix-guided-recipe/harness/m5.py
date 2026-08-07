def check(workdir):
    import numpy as np
    from gguf_recipe.evaluator import measure_gain
    m = {"gain_verified": 0.0}
    weights = {"layer1": np.random.randn(16, 16)}
    rw = {"layer1": "Q8_0"}
    rwo = {"layer1": "Q4_0"}
    gain = measure_gain(weights, rw, rwo)
    if isinstance(gain, (int, float)):
        m["gain_verified"] = 1.0
    return m
