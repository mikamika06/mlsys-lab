import numpy as np


def _row_group_quant_err(X, nbits, group_size):
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
    overhead_bytes = n_rows * g * 2 * 4
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


def choose_kv_budget(
    K: np.ndarray,
    V: np.ndarray,
    candidates: list,
    byte_budget: int,
    group_size: int,
) -> int:
    """
    Enumerate `candidates` = [(nbits_K, nbits_V, R), ...], compute the total
    reconstruction MSE and byte cost of each under per-row-group affine
    min-max quantization (older T-R rows quantized, most recent R rows kept
    exact), discard configs whose byte cost exceeds `byte_budget`, and return
    the index of the feasible config with the smallest MSE (ties -> smaller
    index, i.e. standard argmin order).
    """
    best_idx, best_mse = -1, float("inf")
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    for i, (nk, nv, R) in enumerate(candidates):
        mse, nbytes = _config_cost(K, V, nk, nv, R, group_size)
        if nbytes > byte_budget:
            continue
        if mse < best_mse:
            best_mse, best_idx = mse, i
    return int(best_idx)
