def grade(sol, fx) -> dict:
    import numpy as np
    rng = np.random.default_rng(0)
    d = 64
    W = rng.standard_normal((d, d))
    b = rng.standard_normal(d)

    def block_fn(x):
        return x @ W + b

    x = rng.standard_normal((32, d))

    # reference ratio
    y_ref = x
    for _ in range(12):
        y_ref = block_fn(y_ref)
    ref_ratio = np.linalg.norm(y_ref) / np.linalg.norm(x)

    try:
        cand_ratio = sol.residual_norm_growth(block_fn, x)
    except Exception:
        return {"rel_err": float("inf")}

    rel_err = abs(cand_ratio - ref_ratio) / (abs(ref_ratio) + 1e-12)
    return {"rel_err": rel_err}
