import struct
import numpy as np
from mlsys import scorers


def _oracle(a_bits, b_bits):
    a = np.array([a_bits], dtype=np.uint32).view(np.float32)[0]
    b = np.array([b_bits], dtype=np.uint32).view(np.float32)[0]
    c = np.float32(a + b)
    return int(np.array([c], dtype=np.float32).view(np.uint32)[0])


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)
    cases = [
        (0x3F800000, 0x40000000),
        (0x3F000000, 0x33800000),
        (0x7EFFFFFF, 0x00800000),
        (0x00800000, 0x007FFFFF),
        (0x3F800000, 0xBF800000),
        (0x41200000, 0xC0A00000),
    ]
    for _ in range(200):
        cases.append((
            int(rng.integers(0, 0x7F800000)),
            int(rng.integers(0, 0x7F800000))
        ))

    got = []
    ref = []
    for a, b in cases:
        try:
            got.append(int(sol.fp32_add_bits(a, b)) & 0xFFFFFFFF)
        except Exception:
            return {"byte_exact_fraction": 0.0}
        ref.append(_oracle(a, b))

    got_bytes = np.asarray(got, dtype=np.uint32)
    ref_bytes = np.asarray(ref, dtype=np.uint32)
    return {
        "byte_exact_fraction": scorers.byte_exact_fraction(got_bytes, ref_bytes)
    }
