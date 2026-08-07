import numpy as np


def quantize_q4_k(tensor):
    flat = np.asarray(tensor, dtype=np.float32).flatten()
    n = len(flat)
    blocks = n // 256
    out_bytes = bytearray()
    for b in range(blocks):
        block = flat[b * 256:(b + 1) * 256]
        d = np.max(np.abs(block)) / 7.0
        if d == 0:
            d = 1.0
        q = np.round(block / d).astype(np.int32)
        q = np.clip(q, -8, 7) + 8
        scales = int(round(d * 1000)) & 0xFFFF
        out_bytes.extend(scales.to_bytes(2, 'little'))
        for i in range(0, 256, 2):
            val = (q[i] & 0x0F) | ((q[i + 1] & 0x0F) << 4)
            out_bytes.append(val)
    return bytes(out_bytes)


def dequantize_q4_k(data):
    b_arr = bytearray(data)
    blocks = len(b_arr) // 130
    res = np.zeros(blocks * 256, dtype=np.float32)
    idx = 0
    for b in range(blocks):
        scales = int.from_bytes(b_arr[idx:idx + 2], 'little')
        d = float(scales) / 1000.0
        idx += 2
        for i in range(0, 256, 2):
            val = b_arr[idx]
            idx += 1
            q0 = (val & 0x0F) - 8
            q1 = ((val >> 4) & 0x0F) - 8
            res[b * 256 + i] = q0 * d
            res[b * 256 + i + 1] = q1 * d
    return res
