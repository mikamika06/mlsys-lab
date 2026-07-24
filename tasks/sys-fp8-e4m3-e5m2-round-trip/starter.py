import numpy as np


def fp8_round_trip(x, fmt):
    """Quantize x into the FP8 format named by fmt ("e4m3" or "e5m2") and
    dequantize it back. Use the real hardware-accurate NumPy dtypes exposed
    by the ml_dtypes package: ml_dtypes.float8_e4m3fn for "e4m3" and
    ml_dtypes.float8_e5m2 for "e5m2". Returns a float32 array, same shape as x.
    """
    raise NotImplementedError('your code here')
