import numpy as np


def _oracle_roundtrip(kv):
    arr = np.asarray(kv, dtype=np.float64)
    payload = arr.tobytes()
    restored = np.frombuffer(payload, dtype=arr.dtype).copy()
    return restored.reshape(arr.shape)


def grade(sol, fx) -> dict:
    cases = [
        np.array([0.1234567, -3.1415926, 7.777777], dtype=np.float32),
        np.array(
            [[1.234567, -0.0001234], [99.99991, -42.42424]],
            dtype=np.float32,
        ),
        np.array(
            [[[0.33333334, 0.6666667], [-12.345678, 88.88889]]],
            dtype=np.float32,
        ),
    ]

    worst = 0.0
    for kv in cases:
        try:
            got = np.asarray(sol.serialize_kv_roundtrip(kv), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle_roundtrip(kv)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)

    return {"max_abs_err": worst}
