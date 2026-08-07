def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    import ref
    import numpy as np
    from quant_rec.lowrank import apply_low_rank_corrector

    m = {"low_rank_applied": 0.0, "cost_accounted": 0.0}
    residual = np.random.randn(16, 16)
    approx, cost = apply_low_rank_corrector(residual, 2)
    if isinstance(approx, np.ndarray) and approx.shape == residual.shape:
        m["low_rank_applied"] = 1.0
    if isinstance(cost, float) and cost > 0.0:
        m["cost_accounted"] = 1.0
    return m
