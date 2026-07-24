import numpy as np

from mlsys import scorers


def _oracle(V: np.ndarray, group_size: int, residual_window: int) -> np.ndarray:
    T, d = V.shape
    Tq = T - residual_window
    Vq = V[:Tq]
    Vr = V[Tq:]

    ng = d // group_size
    Vq_g = Vq.reshape(Tq, ng, group_size)

    lo = np.min(Vq_g, axis=-1)
    hi = np.max(Vq_g, axis=-1)
    scale = (hi - lo) / 3.0
    scale = np.where(scale == 0, 1.0, scale)

    code = np.round((Vq_g - lo[:, :, None]) / scale[:, :, None])
    code = np.clip(code, 0, 3)

    Vq_hat = (code * scale[:, :, None] + lo[:, :, None]).reshape(Tq, d)
    return np.concatenate([Vq_hat, Vr], axis=0)


def grade(sol, fx) -> dict:
    group_size = 32
    residual_window = 16
    rng = np.random.default_rng(0)
    V = rng.standard_normal((96, 64)).astype(np.float64)

    ref = _oracle(V, group_size, residual_window)

    try:
        got = sol.kv_int2_residual_window(np.array(V, dtype=np.float64), group_size, residual_window)
        got = np.asarray(got, dtype=np.float64)
        if got.shape != V.shape:
            return {"residual_max_abs_err": float("inf"), "quant_max_abs_err": float("inf")}
    except Exception:
        return {"residual_max_abs_err": float("inf"), "quant_max_abs_err": float("inf")}

    Tq = 96 - residual_window
    residual_err = scorers.max_abs_err(V[Tq:], got[Tq:])
    quant_err = scorers.max_abs_err(ref[:Tq], got[:Tq])

    return {"residual_max_abs_err": residual_err, "quant_max_abs_err": quant_err}
