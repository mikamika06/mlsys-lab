import numpy as np

from mlsys import scorers


def _oracle_dynamic_map(signed=True, max_exponent_bits=7, total_bits=8):
    data = []
    non_sign_bits = total_bits - (1 if signed else 0)
    additional_items = 2 ** (non_sign_bits - max_exponent_bits) - 1
    i = 0
    for i in range(max_exponent_bits):
        if signed:
            fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits) + 1)
        else:
            fraction_items = int(2 ** (i + non_sign_bits - max_exponent_bits + 1) + 1)
        boundaries = np.linspace(0.1, 1, fraction_items)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += list((10.0 ** (-(max_exponent_bits - 1) + i)) * means)
        if signed:
            data += list(-(10.0 ** (-(max_exponent_bits - 1) + i)) * means)
    if additional_items > 0:
        boundaries = np.linspace(0.1, 1, additional_items + 1)
        means = (boundaries[:-1] + boundaries[1:]) / 2.0
        data += list((10.0 ** (-(max_exponent_bits - 1) + i)) * means)
        if signed:
            data += list(-(10.0 ** (-(max_exponent_bits - 1) + i)) * means)
    data.append(0.0)
    data.append(1.0)
    assert len(data) == 2 ** total_bits
    data.sort()
    return np.array(data, dtype=np.float64)


def grade(sol, fx) -> dict:
    ref = _oracle_dynamic_map()

    try:
        got = sol.create_dynamic_map()
    except Exception:
        return {"max_abs_err": float("inf"), "size_ok": 0.0}

    try:
        got = np.asarray(got, dtype=np.float64).ravel()
    except Exception:
        return {"max_abs_err": float("inf"), "size_ok": 0.0}

    if got.shape != ref.shape:
        return {"max_abs_err": float("inf"), "size_ok": 0.0}

    # oracle is sorted ascending and includes an exact 0.0 entry
    size_ok = 1.0 if (np.all(np.diff(got) >= -1e-9) and np.any(np.abs(got) < 1e-9)) else 0.0

    err = scorers.max_abs_err(ref, got)
    if not np.isfinite(err):
        err = float("inf")

    return {"max_abs_err": err, "size_ok": size_ok}
