import struct
import numpy as np


def pack_groupwise_int4(x: np.ndarray) -> bytes:
    x = np.asarray(x, dtype=np.float32)
    out = bytearray()
    for start in range(0, len(x), 8):
        g = x[start:start + 8]
        m = np.max(np.abs(g))
        scale = np.float32(m / np.float32(7.0)) if m != 0 else np.float32(1.0)
        q = np.round(g / scale).astype(np.int32)
        q = np.clip(q, -8, 7)
        n = q + 8
        out.extend(struct.pack("<f", float(scale)))
        for i in range(0, 8, 2):
            out.append(int(n[i]) | (int(n[i + 1]) << 4))
    return bytes(out)
