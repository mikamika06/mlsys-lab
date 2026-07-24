import numpy as np

GROUP_SIZE = 32
BITS = 4
CLIP_RATIOS = np.linspace(1.0, 0.5, 11)


def _oracle(W: np.ndarray, group_size: int, clip_ratios: np.ndarray, bits: int):
    rows, cols = W.shape
    ng = cols // group_size
    Wg = W.reshape(rows, ng, group_size)
    amax = np.max(np.abs(Wg), axis=2)
    qmax = 2 ** (bits - 1) - 1

    n_ratios = clip_ratios.shape[0]
    mse_grid = np.empty((rows, ng, n_ratios), dtype=np.float64)

    for ri in range(n_ratios):
        r = clip_ratios[ri]
        clipped_amax = amax * r
        clipped_amax_safe = np.where(clipped_amax == 0, 1.0, clipped_amax)
        scale = clipped_amax_safe / qmax

        Wc = np.clip(Wg, -clipped_amax[:, :, None], clipped_amax[:, :, None])
        q = np.clip(np.round(Wc / scale[:, :, None]), -qmax, qmax)
        deq = q * scale[:, :, None]

        mse_grid[:, :, ri] = np.mean((Wg - deq) ** 2, axis=2)

    best_idx = np.argmin(mse_grid, axis=2).astype(np.int64)
    best_mse = np.min(mse_grid, axis=2)
    return best_idx, best_mse


def _fail():
    return {"idx_exact_match": 0.0, "mse_max_abs_err": float("inf")}


def grade(sol, fx) -> dict:
    W = fx["clip_w"]
    idx_ref, mse_ref = _oracle(W, GROUP_SIZE, CLIP_RATIOS, BITS)

    try:
        out = sol.awq_clip_search(W.copy(), GROUP_SIZE, CLIP_RATIOS.copy(), BITS)
    except Exception:
        return _fail()

    try:
        idx_got, mse_got = out
        idx_got = np.asarray(idx_got).astype(np.int64)
        mse_got = np.asarray(mse_got, dtype=np.float64)
    except Exception:
        return _fail()

    if idx_got.shape != idx_ref.shape or mse_got.shape != mse_ref.shape:
        return _fail()

    idx_exact_match = float(np.array_equal(idx_got, idx_ref))
    mse_max_abs_err = float(np.max(np.abs(mse_got - mse_ref)))

    return {"idx_exact_match": idx_exact_match, "mse_max_abs_err": mse_max_abs_err}
