import numpy as np

def _gptq_reference(W, X):
    """Reference GPTQ implementation used by the grader."""
    W_mod = W.copy()
    m, n = W.shape
    H = X.T @ X / X.shape[0]  # Hessian approximation
    codes = np.empty_like(W, dtype=np.int8)
    scales = np.zeros(n, dtype=np.float64)

    for j in range(n):
        col = W_mod[:, j]
        scale = np.max(np.abs(col)) / 127.0
        if scale == 0:
            scale = 1.0
        scales[j] = scale

        int_col = np.round(col / scale).astype(np.int8)
        codes[:, j] = int_col

        recon = scale * int_col.astype(np.float64)
        residual = col - recon

        if j + 1 < n:
            factor = H[j, j+1:] / (H[j, j] + 1e-12)  # shape (n-j-1,)
            for k_idx, k in enumerate(range(j + 1, n)):
                W_mod[:, k] += residual * factor[k_idx]

    return codes, scales

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_rel_err = 0.0
    shapes = [(4, 3), (8, 5), (12, 7)]
    for m, n in shapes:
        batch = 20
        W = rng.standard_normal((m, n)).astype(np.float32)
        X = rng.standard_normal((batch, n)).astype(np.float32)

        try:
            codes_cand, scales_cand = sol.gptq_quantize(W, X)
        except Exception:
            return {"rel_err": 1.0}

        codes_ref, scales_ref = _gptq_reference(W, X)

        recon_cand = scales_cand * codes_cand.astype(np.float64)
        recon_ref = scales_ref * codes_ref.astype(np.float64)

        rel_err = np.linalg.norm(recon_cand - recon_ref) / (
            np.linalg.norm(recon_ref) + 1e-12
        )
        if rel_err > max_rel_err:
            max_rel_err = rel_err

    return {"rel_err": max_rel_err}
