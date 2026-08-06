import struct
import numpy as np


def pack_groupwise_int4(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32)
    out = bytearray()
    for start in range(0, len(x), 8):
        g = x[start:start + 8]
        m = np.float32(0.0)
        for val in g:
            abs_val = val if val >= 0.0 else -val
            if abs_val > m:
                m = abs_val
        scale = np.float32(m / np.float32(7.0)) if m != 0.0 else np.float32(1.0)
        n = []
        for val in g:
            r = round(float(np.float32(val) / scale))
            if r < -8:
                r = -8
            elif r > 7:
                r = 7
            n.append(int(r) + 8)
        out.extend(struct.pack("<f", float(scale)))
        for i in range(0, 8, 2):
            out.append(int(n[i]) | (int(n[i + 1]) << 4))
    return bytes(out)
