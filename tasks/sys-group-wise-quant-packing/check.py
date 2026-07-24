import struct
import numpy as np
from mlsys import scorers


def _oracle(x):
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


def grade(sol, fx) -> dict:
    cases = [
        np.array([0, 1, -1, 2, -2, 3, -3, 4], dtype=np.float32),
        np.array([0.5, -0.5, 7.0, -7.0, 3.1, -2.2, 1.0, -1.0], dtype=np.float32),
        np.arange(-16, 16, dtype=np.float32),
        np.zeros(16, dtype=np.float32),
    ]
    score = 1.0
    for x in cases:
        try:
            got = sol.pack_groupwise_int4(x)
        except Exception:
            score = 0.0
            break
        if scorers.byte_exact_fraction(_oracle(x), got) != 1.0:
            score = 0.0
            break
    return {"byte_exact_fraction": score}
