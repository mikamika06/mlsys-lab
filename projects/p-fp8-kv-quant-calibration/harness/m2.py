import numpy as np

def check(workdir):
    from kvquant.quantize import quantize_fp8, dequantize_fp8
    m = {"quant_shape_ok": 0.0, "dequant_error_bounded": 0.0}

    np.random.seed(123)
    x = np.random.randn(16, 64).astype(np.float32)
    scale = float(np.max(np.abs(x)))

    q = quantize_fp8(x, scale)
    dq = dequantize_fp8(q, scale)

    if q.shape == x.shape and q.dtype == np.int8:
        m["quant_shape_ok"] = 1.0

    err = np.max(np.abs(x - dq))
    if err < 0.05:
        m["dequant_error_bounded"] = 1.0

    return m
