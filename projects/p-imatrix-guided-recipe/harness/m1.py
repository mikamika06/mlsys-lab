def check(workdir):
    import numpy as np
    from gguf_recipe.collector import collect_imatrix
    m = {"imatrix_shape_ok": 0.0, "imatrix_non_trivial": 0.0}
    weights = {"layer1": np.random.randn(16, 16)}
    corpus = np.random.randn(10, 16)
    res = collect_imatrix(weights, corpus)
    if "layer1" not in res or res["layer1"].shape != (16, 16):
        return m
    m["imatrix_shape_ok"] = 1.0
    if np.std(res["layer1"]) > 1e-6:
        m["imatrix_non_trivial"] = 1.0
    return m
