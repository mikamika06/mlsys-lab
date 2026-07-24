import numpy as np


def dynamic_quantize_linear(x: np.ndarray) -> dict:
    """ONNX DynamicQuantizeLinear: derive an asymmetric uint8 scale/zero
    point from x's own (0-including) min/max, round-half-to-even, saturate
    to [0, 255].

    Returns {"y": uint8 array, "y_scale": float, "y_zero_point": uint8}.
    """
    raise NotImplementedError('your code here')
