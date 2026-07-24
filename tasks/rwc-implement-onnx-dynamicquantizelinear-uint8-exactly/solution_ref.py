import numpy as np


def dynamic_quantize_linear(x: np.ndarray) -> dict:
    """ONNX DynamicQuantizeLinear: derive an asymmetric uint8 scale/zero
    point from x's own (0-including) min/max, round-half-to-even, saturate
    to [0, 255]."""
    x = np.asarray(x, dtype=np.float64)
    qmin, qmax = 0.0, 255.0

    xmin = min(0.0, float(x.min()))
    xmax = max(0.0, float(x.max()))
    y_scale = (xmax - xmin) / (qmax - qmin)

    intermediate_zp = qmin - xmin / y_scale
    y_zero_point = int(np.clip(np.round(intermediate_zp), qmin, qmax))

    y = np.clip(np.round(x / y_scale) + y_zero_point, qmin, qmax).astype(np.uint8)

    return {
        "y": y,
        "y_scale": float(y_scale),
        "y_zero_point": np.uint8(y_zero_point),
    }
