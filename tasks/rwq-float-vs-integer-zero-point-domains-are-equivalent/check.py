import numpy as np


def _oracle(codes, scale, zp_float):
    codes = np.asarray(codes, dtype=np.float64)
    zp_int_ref = -zp_float / scale
    deq_f_ref = scale * codes + zp_float
    deq_i_ref = scale * (codes - zp_int_ref)
    return zp_int_ref, deq_f_ref, deq_i_ref


def _fail():
    return {
        "max_abs_err": float("inf"),
        "domain_agreement_err": float("inf"),
        "zp_int_err": float("inf"),
    }


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    cases = []
    # hand-picked: zero scale-offset edge, negative/positive bias, non-integer bias
    cases.append((np.array([0, 1, -1, 5, -5, 127, -128], dtype=np.int64), 0.5, -3.0))
    cases.append((np.array([0, 10, 200, 255], dtype=np.int64), 0.0313725, 12.7))
    cases.append((np.arange(-16, 16, dtype=np.int64), 1.7, -0.35))

    # randomized cases
    for _ in range(8):
        n = int(rng.integers(1, 64))
        codes = rng.integers(-128, 128, size=n).astype(np.int64)
        scale = float(rng.uniform(1e-3, 5.0))
        zp_float = float(rng.uniform(-50.0, 50.0))
        cases.append((codes, scale, zp_float))

    max_err = 0.0
    max_domain_err = 0.0
    max_zp_err = 0.0

    for codes, scale, zp_float in cases:
        zp_int_ref, deq_f_ref, deq_i_ref = _oracle(codes, scale, zp_float)

        try:
            out = sol.dual_zero_point_dequant(codes.copy(), scale, zp_float)
        except Exception:
            return _fail()

        try:
            deq_f, deq_i, zp_int = out
            deq_f = np.asarray(deq_f, dtype=np.float64)
            deq_i = np.asarray(deq_i, dtype=np.float64)
            zp_int = float(zp_int)
        except Exception:
            return _fail()

        if deq_f.shape != codes.shape or deq_i.shape != codes.shape:
            return _fail()

        max_err = max(
            max_err,
            float(np.max(np.abs(deq_f - deq_f_ref))),
            float(np.max(np.abs(deq_i - deq_i_ref))),
        )
        max_domain_err = max(max_domain_err, float(np.max(np.abs(deq_f - deq_i))))
        max_zp_err = max(max_zp_err, abs(zp_int - zp_int_ref))

    return {
        "max_abs_err": max_err,
        "domain_agreement_err": max_domain_err,
        "zp_int_err": max_zp_err,
    }
