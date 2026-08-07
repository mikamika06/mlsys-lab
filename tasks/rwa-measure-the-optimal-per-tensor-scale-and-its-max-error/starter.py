def optimal_scale_and_error(x: list[float], fmt: str) -> tuple[float, float]:
    """
    Compute the standard per-tensor calibration scale for FP8 format `fmt`
    ("e4m3" or "e5m2"): scale = amax(|x|) / FORMAT_MAX[fmt]. Quantize
    x/scale to the nearest representable minifloat value (clamped at
    +-FORMAT_MAX), dequantize by multiplying back by `scale`, and return
    (scale, max_abs_dequant_error).
    """
    raise NotImplementedError('your code here')
