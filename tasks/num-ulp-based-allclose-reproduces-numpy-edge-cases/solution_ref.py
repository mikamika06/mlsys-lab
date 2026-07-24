import numpy as np


def ulp_allclose_report(a, b, max_ulps, atol):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    diff = np.abs(a - b)
    spacing = np.maximum(
        np.abs(np.spacing(np.abs(a))),
        np.abs(np.spacing(np.abs(b))),
    )
    ulp_ok = np.logical_or(diff <= max_ulps * spacing, a == b)
    atol_ok = diff <= atol

    return ulp_ok.astype(bool), atol_ok.astype(bool)
