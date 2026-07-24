import numpy as np


def _reference(a, b, max_ulps, atol):
    diff = np.abs(a - b)
    spacing = np.maximum(np.abs(np.spacing(np.abs(a))), np.abs(np.spacing(np.abs(b))))
    ulp_ok = diff <= (max_ulps * spacing)
    ulp_ok = np.logical_or(ulp_ok, a == b)
    atol_ok = diff <= atol
    return ulp_ok, atol_ok


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([0.0, 1.0, 1e-300, 1e20, -1e20], dtype=np.float64),
            np.array([
                np.nextafter(0.0, np.inf),
                np.nextafter(1.0, np.inf),
                2e-300,
                1e20 + 1e6,
                -1e20 - 1e6,
            ], dtype=np.float64),
            2,
            1e-8,
        ),
        (
            np.array([1.0, 1000000.0, -0.0, 3.141592653589793], dtype=np.float64),
            np.array([
                np.nextafter(np.nextafter(1.0, np.inf), np.inf),
                np.nextafter(1000000.0, np.inf),
                0.0,
                3.141592653589794,
            ], dtype=np.float64),
            3,
            1e-12,
        ),
        (
            np.array([np.finfo(np.float64).tiny, 1e-10, -1e10], dtype=np.float64),
            np.array([0.0, 1.0000000000000001e-10, -1e10 + 1.0], dtype=np.float64),
            1,
            1e-12,
        ),
    ]

    ok = 1.0
    for a, b, max_ulps, atol in cases:
        try:
            got = sol.ulp_allclose_report(a, b, max_ulps, atol)
            got = (np.asarray(got[0]), np.asarray(got[1]))
        except Exception:
            ok = 0.0
            break
        ref = _reference(a, b, max_ulps, atol)
        if (
            got[0].dtype != np.bool_
            or got[1].dtype != np.bool_
            or not np.array_equal(got[0], ref[0])
            or not np.array_equal(got[1], ref[1])
        ):
            ok = 0.0
            break

    return {"exact_match": ok}
