import numpy as np

def _ref(expr):
    if expr == "inf-inf":
        val = np.inf - np.inf
    elif expr == "0*inf":
        val = 0.0 * np.inf
    elif expr == "nan==nan":
        val = float('nan') == float('nan')
    elif expr.startswith("min(nan,"):
        x_str = expr.split(",")[1].rstrip(")")
        x = float(x_str)
        val = np.minimum(np.nan, x)  # propagate NaN
    elif expr == "inf/inf":
        val = np.inf / np.inf
    else:
        raise ValueError(f"Unsupported expression: {expr}")
    return int(np.asarray(val, dtype=np.float64).view(np.uint64))

def grade(sol, fx) -> dict:
    cases = [
        "inf-inf",
        "0*inf",
        "nan==nan",
        "min(nan,3.14)",
        "inf/inf"
    ]
    ok = 1.0
    for expr in cases:
        try:
            got = sol.predict_special_value(expr)
            exp = _ref(expr)
        except Exception:
            ok = 0.0
            break
        if got != exp:
            ok = 0.0
            break
    return {"exact_match": ok}
