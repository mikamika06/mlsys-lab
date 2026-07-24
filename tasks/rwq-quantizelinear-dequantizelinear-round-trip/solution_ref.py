import numpy as np


def qdq_round_trip(x: np.ndarray, scale: float, zero_point: int):
    """ONNX QuantizeLinear / DequantizeLinear round-trip.

    q   = clip(round(x / scale) + zero_point, 0, 255)   (uint8)
    deq = (q - zero_point) * scale
    """
    x = np.asarray(x, dtype=np.float64)
    q = np.clip(np.round(x / scale) + zero_point, 0, 255).astype(np.uint8)
    deq = (q.astype(np.float64) - zero_point) * scale
    return q, deq
