import numpy as np

GROUP_SIZE = 32


def _sym_int4_dequant(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale_safe = np.where(scale == 0, 1.0, scale)
    code = np.clip(np.round(x / scale_safe), -7, 7)
    return code * scale_safe


def _per_axis_mse(W: np.ndarray) -> float:
    amax = np.max(np.abs(W), axis=1)
    scale = amax / 7.0
    deq = _sym_int4_dequant(W, scale[:, None])
    return float(np.mean((W - deq) ** 2))


def _per_group_mse(W: np.ndarray, group_size: int) -> float:
    rows, cols = W.shape
    ng = cols // group_size
    Wg = W.reshape(rows, ng, group_size)
    amax = np.max(np.abs(Wg), axis=2)
    scale = amax / 7.0
    deq = _sym_int4_dequant(Wg, scale[:, :, None]).reshape(rows, cols)
    return float(np.mean((W - deq) ** 2))


def _fail():
    return {
        "mse_per_axis_err": float("inf"),
        "mse_per_group_err": float("inf"),
        "finer_grain_wins": 0.0,
    }


def grade(sol, fx) -> dict:
    W = fx["qnt_w"]
    mse_axis_ref = _per_axis_mse(W)
    mse_group_ref = _per_group_mse(W, GROUP_SIZE)

    try:
        out = sol.qint4_granularity_mse(W.copy(), GROUP_SIZE)
    except Exception:
        return _fail()

    try:
        mse_axis_got = float(out[0])
        mse_group_got = float(out[1])
    except Exception:
        return _fail()

    if not (np.isfinite(mse_axis_got) and np.isfinite(mse_group_got)):
        return _fail()

    return {
        "mse_per_axis_err": abs(mse_axis_got - mse_axis_ref),
        "mse_per_group_err": abs(mse_group_got - mse_group_ref),
        "finer_grain_wins": 1.0 if mse_group_got < mse_axis_got else 0.0,
    }
