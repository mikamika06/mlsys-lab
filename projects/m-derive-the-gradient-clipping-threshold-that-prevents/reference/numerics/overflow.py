import numpy as np

def check_precision_overflow(value, dtype="float16"):
    if dtype == "float16":
        if abs(value) > 65504.0:
            return float('inf')
    elif dtype == "bfloat16":
        if abs(value) > 3.38e38:
            return float('inf')
    return float(value)
