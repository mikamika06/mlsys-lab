import numpy as np

def _to_bits(val):
    return int(np.asarray(val, dtype=np.float64).view(np.uint64))

def predict_special_value(expr: str) -> int:
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
    return _to_bits(val)
