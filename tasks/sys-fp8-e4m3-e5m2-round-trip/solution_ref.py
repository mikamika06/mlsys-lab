import numpy as np
import ml_dtypes

_DTYPES = {"e4m3": ml_dtypes.float8_e4m3fn, "e5m2": ml_dtypes.float8_e5m2}


def fp8_round_trip(x, fmt):
    x = np.asarray(x, dtype=np.float32)
    dt = _DTYPES[fmt]
    return x.astype(dt).astype(np.float32)
