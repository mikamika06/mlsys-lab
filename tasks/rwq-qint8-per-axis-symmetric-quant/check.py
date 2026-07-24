import numpy as np


def _oracle(W, axis):
    W = np.asarray(W, dtype=np.float64)
    reduce_axes = tuple(a for a in range(W.ndim) if a != axis)
    absmax = np.max(np.abs(W), axis=reduce_axes, keepdims=True)
    absmax = np.where(absmax == 0.0, 1.0, absmax)
    scale = absmax / 127.0
    codes = np.clip(np.round(W / scale), -127, 127)
    deq = codes * scale
    return codes, scale, deq


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    worst_abs = 0.0

    for axis in (0, 1):
        for _ in range(3):
            d0 = int(rng.integers(3, 8))
            d1 = int(rng.integers(3, 8))
            W = rng.normal(size=(d0, d1)) * rng.uniform(0.2, 3.0, size=(d0, 1))
            # give one channel a zero row/col so the absmax==0 guard is exercised
            zero_idx = int(rng.integers(0, W.shape[axis]))
            if axis == 0:
                W[zero_idx, :] = 0.0
            else:
                W[:, zero_idx] = 0.0

            codes_exp, scale_exp, deq_exp = _oracle(W, axis)

            try:
                codes_got, scale_got, deq_got = sol.per_axis_qint8(W.copy(), axis)
                codes_got = np.asarray(codes_got, dtype=np.float64)
                scale_got = np.asarray(scale_got, dtype=np.float64)
                deq_got = np.asarray(deq_got, dtype=np.float64)
            except Exception:
                ok = 0.0
                worst_abs = float("inf")
                continue

            if codes_got.shape != codes_exp.shape or not np.array_equal(codes_got, codes_exp):
                ok = 0.0

            if deq_got.shape != deq_exp.shape:
                worst_abs = float("inf")
                continue

            worst_abs = max(worst_abs, float(np.max(np.abs(deq_got - deq_exp))))

    return {"exact_match": ok, "max_abs_err": worst_abs}
