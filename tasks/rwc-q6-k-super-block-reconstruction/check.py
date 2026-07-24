import numpy as np


def _oracle_dequantize(d, scales, ql, qh):
    d = float(d)
    scales = [int(v) for v in np.asarray(scales, dtype=np.int8)]
    ql = [int(v) for v in np.asarray(ql, dtype=np.uint8)]
    qh = [int(v) for v in np.asarray(qh, dtype=np.uint8)]

    y = [0.0] * 256

    for half in range(2):
        y_off = half * 128
        ql_off = half * 64
        qh_off = half * 32
        sc_off = half * 8

        for l in range(32):
            is_ = l // 16
            q1 = ((ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4)) - 32
            q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) - 32
            q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) - 32
            q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) - 32

            y[y_off + l] = d * scales[sc_off + is_] * q1
            y[y_off + l + 32] = d * scales[sc_off + is_ + 2] * q2
            y[y_off + l + 64] = d * scales[sc_off + is_ + 4] * q3
            y[y_off + l + 96] = d * scales[sc_off + is_ + 6] * q4

    return np.asarray(y, dtype=np.float32)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    d = float(rng.uniform(0.001, 2.0))
    scales = rng.integers(-128, 128, size=16).astype(np.int8)
    ql = rng.integers(0, 256, size=128).astype(np.uint8)
    qh = rng.integers(0, 256, size=64).astype(np.uint8)

    ref = _oracle_dequantize(d, scales, ql, qh)

    try:
        got = sol.q6_k_dequantize(d, scales, ql, qh)
        got = np.asarray(got, dtype=np.float64)
        if got.shape != (256,):
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got - ref.astype(np.float64))))
    except Exception:
        return {"max_abs_err": float("inf")}

    return {"max_abs_err": err}
