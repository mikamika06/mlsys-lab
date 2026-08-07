def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    import numpy as np
    from quant_rec.rounding import optimize_rounding

    m = {"rounding_optimized": 0.0}
    _, quantized, _, grid = ref.get_test_data()
    res = optimize_rounding(quantized, grid)
    if isinstance(res, np.ndarray) and res.shape == quantized.shape:
        m["rounding_optimized"] = 1.0
    return m
