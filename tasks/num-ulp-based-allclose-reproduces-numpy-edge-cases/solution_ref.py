import numpy as np


def ulp_allclose_report(a, b, max_ulps, atol):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n = a.shape[0]
    ulp_ok_list = [False] * n
    atol_ok_list = [False] * n

    for i in range(n):
        val_a = a[i]
        val_b = b[i]

        diff = val_a - val_b
        if diff < 0.0:
            diff = -diff

        abs_a = val_a if val_a >= 0.0 else -val_a
        abs_b = val_b if val_b >= 0.0 else -val_b

        sp_a = np.spacing(abs_a)
        if sp_a < 0.0:
            sp_a = -sp_a

        sp_b = np.spacing(abs_b)
        if sp_b < 0.0:
            sp_b = -sp_b

        spacing = sp_a if sp_a >= sp_b else sp_b

        limit = max_ulps * spacing
        is_ulp_ok = (diff <= limit) or (val_a == val_b)
        is_atol_ok = diff <= atol

        ulp_ok_list[i] = bool(is_ulp_ok)
        atol_ok_list[i] = bool(is_atol_ok)

    return np.array(ulp_ok_list, dtype=bool), np.array(atol_ok_list, dtype=bool)
