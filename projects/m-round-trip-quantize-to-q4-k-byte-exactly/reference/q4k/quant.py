import numpy as np


def quantize_q4_k(weights: np.ndarray) -> bytes:
    w = weights.astype(np.float32)
    n = len(w)
    assert n == 256
    scales = np.zeros(8, dtype=np.float32)
    mins = np.zeros(8, dtype=np.float32)
    subblocks = w.reshape(8, 32)
    for i in range(8):
        sb = subblocks[i]
        mx = np.max(sb)
        mn = np.min(sb)
        if mx == mn:
            scales[i] = 0.0
            mins[i] = mn
        else:
            scales[i] = (mx - mn) / 15.0
            mins[i] = mn

    d = np.max(scales) if np.max(scales) > 0 else 1.0
    dmin = np.min(mins) if np.min(mins) < 0 else 0.0

    sc_byte = int(np.clip(np.round(d * 127.0), 0, 255))
    m_byte = int(np.clip(np.round(dmin * 127.0), 0, 255))

    quantized = np.zeros((8, 16), dtype=np.uint8)
    for i in range(8):
        sb = subblocks[i]
        scale_val = scales[i] if scales[i] > 0 else 1.0
        q = np.round((sb - mins[i]) / scale_val)
        q = np.clip(q, 0, 15).astype(np.uint8)
        for j in range(16):
            quantized[i, j] = q[2 * j] | (q[2 * j + 1] << 4)

    header = bytes([sc_byte, m_byte])
    body = quantized.tobytes()
    return header + body


def dequantize_q4_k(data: bytes) -> np.ndarray:
    header = data[:2]
    body = data[2:]
    sc_byte = header[0]
    m_byte = header[1]
    d = sc_byte / 127.0
    dmin = m_byte / 127.0

    quantized = np.frombuffer(body, dtype=np.uint8).reshape(8, 16)
    out = np.zeros(256, dtype=np.float32)
    for i in range(8):
        for j in range(16):
            b = quantized[i, j]
            q0 = b & 0x0F
            q1 = (b >> 4) & 0x0F
            out[i * 32 + 2 * j] = q0 * d + dmin
            out[i * 32 + 2 * j + 1] = q1 * d + dmin
    return out
