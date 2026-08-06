import numpy as np

def _to_bits(val):
    return int(np.asarray(val, dtype=np.float64).view(np.uint64))

def predict_special_value(expr: str) -> int:
    if expr == "inf-inf":
        val = float('inf') - float('inf')
    elif expr == "0*inf":
        val = 0.0 * float('inf')
    elif expr == "nan==nan":
        val = float('nan') == float('nan')
    elif expr.startswith("min(nan,"):
        x_str = expr.split(",")[1].rstrip(")")
        x = float(x_str)
        nan_val = float('nan')
        val = nan_val if (nan_val != nan_val) else (x if x < nan_val else nan_val)
        if x != x:
            val = x
    elif expr == "inf/inf":
        val = float('inf') / float('inf')
    else:
        raise ValueError(f"Unsupported expression: {expr}")
    return _to_bits(val)
