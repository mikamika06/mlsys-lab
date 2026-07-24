import numpy as np


def _oracle(x, scale, zero_point):
    x = np.asarray(x, dtype=np.float64)
    q = np.clip(np.round(x / scale) + zero_point, 0, 255).astype(np.uint8)
    deq = (q.astype(np.float64) - zero_point) * scale
    return q, deq


def _build_cases():
    cases = []
    for seed, n, scale, zp in [(0, 200, 0.05, 128), (1, 100, 1.5, 10), (2, 150, 0.01, 250)]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-20.0, 20.0, size=n)
        cases.append((x, scale, zp))
    return cases


def grade(sol, fx) -> dict:
    codes_ok = 1.0
    worst_deq_err = 0.0

    for x, scale, zp in _build_cases():
        q_ref, deq_ref = _oracle(x, scale, zp)

        try:
            got = sol.qdq_round_trip(x.copy(), scale, zp)
            q_got, deq_got = got
            q_got = np.asarray(q_got)
            deq_got = np.asarray(deq_got, dtype=np.float64)
        except Exception:
            return {"codes_exact_match": 0.0, "dequant_max_abs_err": float("inf")}

        if q_got.shape != q_ref.shape or deq_got.shape != deq_ref.shape:
            return {"codes_exact_match": 0.0, "dequant_max_abs_err": float("inf")}
        if not np.all(np.isfinite(deq_got)):
            return {"codes_exact_match": 0.0, "dequant_max_abs_err": float("inf")}

        if not np.array_equal(q_got.astype(np.int64), q_ref.astype(np.int64)):
            codes_ok = 0.0

        worst_deq_err = max(worst_deq_err, float(np.max(np.abs(deq_got - deq_ref))))

    return {"codes_exact_match": codes_ok, "dequant_max_abs_err": worst_deq_err}
