import numpy as np

def quantize_q4_k(x):
    x = np.asarray(x, dtype=np.float32)
    assert x.size == 256
    sub = x.reshape(8, 32)
    mins = sub.min(axis=1)
    maxs = sub.max(axis=1)
    scales = (maxs - mins) / 15.0
    scales = np.clip(scales, 1e-5, None)
    scales_bytes = scales.astype(np.float32).tobytes()
    mins_bytes = mins.astype(np.float32).tobytes()
    quants = []
    for i in range(8):
        s = scales[i]
        m = mins[i]
        q = np.round((sub[i] - m) / s).astype(np.uint8)
        q = np.clip(q, 0, 15)
        quants.append(q)
    q_arr = np.array(quants, dtype=np.uint8)
    packed_quants = np.zeros(128, dtype=np.uint8)
    for i in range(128):
        low = q_arr.flatten()[2 * i]
        high = q_arr.flatten()[2 * i + 1]
        packed_quants[i] = (high << 4) | low
    return scales_bytes + mins_bytes + packed_quants.tobytes()

def dequantize_q4_k(b):
    b = bytes(b)
    scales = np.frombuffer(b[0:32], dtype=np.float32).copy()
    mins = np.frombuffer(b[32:64], dtype=np.float32).copy()
    packed = np.frombuffer(b[64:], dtype=np.uint8)
    q_flat = np.zeros(256, dtype=np.uint8)
    for i in range(128):
        val = packed[i]
        q_flat[2 * i] = val & 0x0F
        q_flat[2 * i + 1] = (val >> 4) & 0x0F
    q_arr = q_flat.reshape(8, 32)
    out = np.zeros((8, 32), dtype=np.float32)
    for i in range(8):
        out[i] = q_arr[i].astype(np.float32) * scales[i] + mins[i]
    return out.flatten()

def round_trip_q4_k(x):
    b = quantize_q4_k(x)
    back = dequantize_q4_k(b)
    b_round = quantize_q4_k(back)
    return 1.0 if b == b_round else 0.0
