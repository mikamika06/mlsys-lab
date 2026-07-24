import numpy as np


def _fp_values(exp_bits, mant_bits, bias):
    vals = []
    for e in range(1 << exp_bits):
        for m in range(1 << mant_bits):
            if e == 0:
                if m == 0:
                    v = 0.0
                else:
                    v = m * (2.0 ** (1 - bias - mant_bits))
            else:
                v = (1.0 + m / (2 ** mant_bits)) * (2.0 ** (e - bias))
            vals.append(v)
    return np.unique(np.array(vals, dtype=np.float64))


def _fp8_quant(x, scale, exp_bits, mant_bits, bias):
    levels = _fp_values(exp_bits, mant_bits, bias)
    y = np.abs(x / scale)
    idx = np.searchsorted(levels, y)
    idx = np.clip(idx, 1, len(levels) - 1)
    left = levels[idx - 1]
    right = levels[idx]
    q = np.where(y - left <= right - y, left, right)
    q = np.sign(x) * q
    return q * scale


def _int8_quant(x, scale):
    q = np.clip(np.rint(x / scale), -127, 127)
    return q * scale


def _rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def _oracle_one(x, kind):
    if kind == "e4m3":
        fn = lambda s: _fp8_quant(x, s, 4, 3, 7)
        max_value = np.max(_fp_values(4, 3, 7))
    elif kind == "e5m2":
        fn = lambda s: _fp8_quant(x, s, 5, 2, 15)
        max_value = np.max(_fp_values(5, 2, 15))
    else:
        fn = lambda s: _int8_quant(x, s)
        max_value = 127.0

    base = np.max(np.abs(x)) / max_value
    best = None
    for multiplier in np.geomspace(0.25, 4.0, 41):
        scale = base * multiplier
        if scale > 0:
            err = _rel_err(x, fn(scale))
            if best is None or err < best:
                best = err
    return float(best)


def _oracle(x):
    out = {
        "e4m3_error": _oracle_one(x, "e4m3"),
        "e5m2_error": _oracle_one(x, "e5m2"),
        "int8_error": _oracle_one(x, "int8"),
    }
    names = {
        "e4m3": out["e4m3_error"],
        "e5m2": out["e5m2_error"],
        "int8": out["int8_error"],
    }
    out["best_format"] = min(names, key=names.get)
    return out


def grade(sol, fx) -> dict:
    cases = [
        np.array([0.1, -0.2, 0.3, 12.0], dtype=np.float64),
        np.array([-1.0, 0.5, 2.0, 4.0, 64.0], dtype=np.float64),
        np.array([0.001, -0.002, 0.004, 50.0], dtype=np.float64),
        np.array([1.5, -2.5, 3.5, 7.5, 15.5], dtype=np.float64),
    ]

    ref_values = []
    got_values = []
    best_ok = 1.0

    for x in cases:
        ref = _oracle(x)
        try:
            got = sol.compare_quant_formats(x)
        except Exception:
            return {"rel_err": 1.0, "best_match": 0.0}

        for key in ["e4m3_error", "e5m2_error", "int8_error"]:
            ref_values.append(ref[key])
            got_values.append(float(got[key]))
        if got.get("best_format") != ref["best_format"]:
            best_ok = 0.0

    return {
        "rel_err": _rel_err(np.array(ref_values), np.array(got_values)),
        "best_match": best_ok,
    }
