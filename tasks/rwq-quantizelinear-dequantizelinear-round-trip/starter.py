import numpy as np


def qdq_round_trip(x: np.ndarray, scale: float, zero_point: int):
    """ONNX QuantizeLinear / DequantizeLinear round-trip.

    x: float64 array, any shape.
    scale: positive float.
    zero_point: int in [0, 255].

    q   = clip(round(x / scale) + zero_point, 0, 255), cast to uint8
          (round-half-to-even, i.e. plain np.round).
    deq = (q.astype(float64) - zero_point) * scale

    Returns (q, deq).
    """
    raise NotImplementedError('your code here')
