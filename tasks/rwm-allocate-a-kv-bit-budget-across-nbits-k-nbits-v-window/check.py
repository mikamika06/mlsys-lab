import numpy as np


def _row_group_quant_err(X, nbits, group_size):
    """Sum-of-squared reconstruction error + element count for per-row-group
    affine min-max quantization of X (n_rows, d)."""
    n, d = X.shape
    if n == 0:
        return 0.0, 0
    g = d // group_size
    Xr = X.reshape(n, g, group_size).astype(np.float64)
    mn = Xr.min(axis=2, keepdims=True)
    mx = Xr.max(axis=2, keepdims=True)
    levels = (1 << nbits) - 1
    scale = (mx - mn) / levels
    scale = np.where(scale == 0, 1.0, scale)
    codes = np.clip(np.round((Xr - mn) / scale), 0, levels)
    recon = codes * scale + mn
    err_sq = float(np.sum((recon - Xr) ** 2))
    return err_sq, int(Xr.size)


def _bytes_for_quant(n_rows, dim, nbits, group_size):
    if n_rows == 0:
        return 0
    g = dim // group_size
    code_bits = n_rows * g * group_size * nbits
    code_bytes = int(np.ceil(code_bits / 8.0))
    overhead_bytes = n_rows * g * 2 * 4  # fp32 scale + fp32 min per group per row
    return code_bytes + overhead_bytes


def _config_cost(K, V, nbits_K, nbits_V, R, group_size):
    T, d = K.shape
    R = min(R, T)
    old = T - R
    Kold, Kwin = K[:old], K[old:]
    Vold, Vwin = V[:old], V[old:]
    errK, cntK = _row_group_quant_err(Kold, nbits_K, group_size)
    errV, cntV = _row_group_quant_err(Vold, nbits_V, group_size)
    win_cnt = Kwin.size + Vwin.size
    total_cnt = cntK + cntV + win_cnt
    mse = (errK + errV) / total_cnt if total_cnt > 0 else 0.0
    nbytes = (
        _bytes_for_quant(old, d, nbits_K, group_size)
        + _bytes_for_quant(old, d, nbits_V, group_size)
        + Kwin.size * 4
        + Vwin.size * 4
    )
    return mse, nbytes


def _oracle_choose(K, V, candidates, byte_budget, group_size):
    best_idx, best_mse = -1, float("inf")
    for i, (nk, nv, R) in enumerate(candidates):
        mse, nbytes = _config_cost(K, V, nk, nv, R, group_size)
        if nbytes > byte_budget:
            continue
        if mse < best_mse:
            best_mse, best_idx = mse, i
    return best_idx


def grade(sol, fx) -> dict:
    """
    The grader builds several random (K, V) KV-cache pairs and small candidate
    grids of (nbits_K, nbits_V, R), enumerates costs/MSE with a NumPy oracle,
    and checks that the submission returns the exact same argmin index over
    the feasible (byte-budget-respecting) subset.
    """
    rng = np.random.default_rng(0)
    ok = 1.0
    for _ in range(8):
        try:
            T = int(rng.integers(8, 24))
            dim = int(rng.choice([4, 8]))
            group_size = dim
            K = (rng.normal(size=(T, dim)) * rng.uniform(0.5, 3.0)).astype(np.float64)
            V = (rng.normal(size=(T, dim)) * rng.uniform(0.5, 3.0)).astype(np.float64)

            bit_opts = [2, 3, 4]
            win_opts = sorted(set([0, T // 4, T // 2]))
            candidates = [
                (nk, nv, R) for nk in bit_opts for nv in bit_opts for R in win_opts
            ]

            cheapest_bytes = min(
                _config_cost(K, V, nk, nv, R, group_size)[1] for nk, nv, R in candidates
            )
            full_fp32_bytes = T * dim * 4 * 2
            budget = int(rng.uniform(cheapest_bytes * 1.05, max(full_fp32_bytes * 0.6, cheapest_bytes * 1.1)))
            budget = max(budget, cheapest_bytes)

            expected_idx = _oracle_choose(K, V, candidates, budget, group_size)
            got_idx = sol.choose_kv_budget(K.copy(), V.copy(), list(candidates), int(budget), group_size)
        except Exception:
            ok = 0.0
            break

        if expected_idx < 0 or int(got_idx) != expected_idx:
            ok = 0.0
            break
    return {"argmin_index": ok}
