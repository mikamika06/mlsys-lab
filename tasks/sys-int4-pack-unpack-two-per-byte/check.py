import numpy as np
from mlsys.scorers import byte_exact_fraction

def _ref_pack(values):
    n = values.size
    padded_len = n + (n % 2)
    padded = np.zeros(padded_len, dtype=np.uint8)   # zeros, not empty: the pad nibble must be defined
    padded[:n] = values
    if n % 2:
        padded[-1] &= 0xF0
    high = (padded[::2] << 4) & 0xF0
    low = padded[1::2]
    return high | low

def _ref_unpack(packed, length):
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    unpacked = np.zeros(length + (length % 2), dtype=np.uint8)
    unpacked[::2] = high[:len(high)]
    unpacked[1::2] = low[:len(low)]
    return unpacked[:length]

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    byte_ok = 1.0
    roundtrip_ok = 1.0
    for _ in range(5):
        n = rng.integers(1, 50)
        vals = rng.integers(0, 16, size=n, dtype=np.uint8)
        try:
            packed = sol.pack_int4(vals)
            ref_packed = _ref_pack(vals)
            if byte_exact_fraction(ref_packed.tobytes(), packed.tobytes()) < 1.0 - 1e-12:
                byte_ok = 0.0
                break
            unpacked = sol.unpack_int4(packed, n)
            if not np.array_equal(unpacked, vals):
                roundtrip_ok = 0.0
                break
        except Exception:
            return {"byte_exact_fraction": 0.0, "exact_match": 0.0}
    return {"byte_exact_fraction": byte_ok, "exact_match": roundtrip_ok}
