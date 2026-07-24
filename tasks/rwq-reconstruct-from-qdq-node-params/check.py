import numpy as np


def _oracle(q, scale, zero_point, axis):
    q = np.asarray(q, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    zp = np.asarray(zero_point, dtype=np.float64)

    if scale.ndim == 0:
        s, z = scale, zp
    else:
        shape = [1] * q.ndim
        shape[axis] = -1
        s = scale.reshape(shape)
        z = zp.reshape(shape)

    return (q - z) * s


def _build_cases(fx):
    rng = np.random.default_rng(1)
    cases = []

    # fixture: uint8 codes, per-axis-0 scale/zero_point (matches an ONNX
    # per-channel QDQ weight node)
    cases.append((fx["qdq_q"], fx["qdq_scale"], fx["qdq_zp"], 0))

    # scalar (per-tensor) case
    q = rng.integers(0, 256, size=(5, 7)).astype(np.uint8)
    cases.append((q, 0.0137, 128, 0))

    # per-axis on the last axis
    q = rng.integers(0, 256, size=(4, 9)).astype(np.uint8)
    scale = rng.uniform(0.001, 0.3, size=9)
    zp = rng.integers(0, 256, size=9).astype(np.uint8)
    cases.append((q, scale, zp, 1))

    # a few more randomized per-axis-0 cases
    for _ in range(4):
        n0 = int(rng.integers(2, 8))
        n1 = int(rng.integers(2, 12))
        q = rng.integers(0, 256, size=(n0, n1)).astype(np.uint8)
        scale = rng.uniform(0.0005, 1.0, size=n0)
        zp = rng.integers(0, 256, size=n0).astype(np.uint8)
        cases.append((q, scale, zp, 0))

    return cases


def grade(sol, fx) -> dict:
    max_err = 0.0

    for q, scale, zp, axis in _build_cases(fx):
        ref = _oracle(q, scale, zp, axis)

        try:
            got = np.asarray(
                sol.dequantize_linear(np.asarray(q).copy(), scale, zp, axis),
                dtype=np.float64,
            )
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        max_err = max(max_err, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": max_err}
