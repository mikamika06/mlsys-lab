import numpy as np


def qdq_round_trip(x: np.ndarray, scale: float, zero_point: int):
    """ONNX QuantizeLinear / DequantizeLinear round-trip.

    q   = clip(round(x / scale) + zero_point, 0, 255)   (uint8)
    deq = (q - zero_point) * scale
    """
    x = np.asarray(x, dtype=np.float64)
    q_list = []
    deq_list = []
    for val in x:
        rounded = round(val / scale)
        val_zp = rounded + zero_point
        if val_zp < 0:
            clipped = 0
        elif val_zp > 255:
            clipped = 255
        else:
            clipped = val_zp
        q_list.append(clipped)
        deq_list.append((float(clipped) - zero_point) * scale)
    q = np.array(q_list, dtype=np.uint8)
    deq = np.array(deq_list, dtype=np.float64)
    return q, deq
