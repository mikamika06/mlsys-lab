import numpy as np

GROUP_SIZE = 128


def _oracle(W: np.ndarray, group_size: int):
    rows, cols = W.shape
    ng = cols // group_size
    Wg = W.reshape(rows, ng, group_size)

    gmin = np.min(Wg, axis=2)
    gmax = np.max(Wg, axis=2)
    scale = (gmax - gmin) / 15.0
    scale_safe = np.where(scale == 0, 1.0, scale)
    zp = gmin

    q = np.round((Wg - zp[:, :, None]) / scale_safe[:, :, None])
    q = np.clip(q, 0, 15).astype(np.int64)
    deq = q.astype(np.float64) * scale_safe[:, :, None] + zp[:, :, None]

    return (
        q.reshape(rows, cols),
        scale,
        zp,
        deq.reshape(rows, cols),
    )


def _fail():
    return {
        "codes_exact_match": 0.0,
        "params_max_abs_err": float("inf"),
        "max_abs_err": float("inf"),
    }


def grade(sol, fx) -> dict:
    W = fx["tao_w"]
    q_ref, scale_ref, zp_ref, deq_ref = _oracle(W, GROUP_SIZE)

    try:
        out = sol.tinygemm_int4_quantize(W.copy(), GROUP_SIZE)
    except Exception:
        return _fail()

    try:
        q_got, scale_got, zp_got, deq_got = out
        q_got = np.asarray(q_got).astype(np.int64)
        scale_got = np.asarray(scale_got, dtype=np.float64)
        zp_got = np.asarray(zp_got, dtype=np.float64)
        deq_got = np.asarray(deq_got, dtype=np.float64)
    except Exception:
        return _fail()

    if (
        q_got.shape != q_ref.shape
        or scale_got.shape != scale_ref.shape
        or zp_got.shape != zp_ref.shape
        or deq_got.shape != deq_ref.shape
    ):
        return _fail()

    if np.any(q_got < 0) or np.any(q_got > 15):
        return _fail()

    codes_exact_match = float(np.array_equal(q_got, q_ref))
    params_max_abs_err = float(
        max(
            np.max(np.abs(scale_got - scale_ref)),
            np.max(np.abs(zp_got - zp_ref)),
        )
    )
    max_abs_err = float(np.max(np.abs(deq_got - deq_ref)))

    return {
        "codes_exact_match": codes_exact_match,
        "params_max_abs_err": params_max_abs_err,
        "max_abs_err": max_abs_err,
    }
