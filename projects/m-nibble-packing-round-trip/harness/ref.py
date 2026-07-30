import numpy as np

CODEBOOK = np.array([
    -1.0, -0.84, -0.68, -0.52,
    -0.36, -0.20, -0.04, 0.0,
    0.04, 0.12, 0.20, 0.28,
    0.40, 0.55, 0.75, 1.0,
], dtype=np.float64)


def pack_nibbles(codes):
    codes = [int(c) & 0x0F for c in codes]
    n = len(codes)
    nbytes = (n + 1) // 2
    packed = [0] * nbytes
    for i in range(nbytes):
        lo = codes[2 * i]
        hi = codes[2 * i + 1] if 2 * i + 1 < n else 0
        packed[i] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    return np.array(packed, dtype=np.uint8)


def unpack_nibbles(packed, n):
    packed = [int(b) for b in packed]
    codes = [0] * n
    for i in range(n):
        byte = packed[i // 2]
        codes[i] = (byte & 0x0F) if i % 2 == 0 else ((byte >> 4) & 0x0F)
    return np.array(codes, dtype=np.uint8)


def dequantize_block(packed, n, absmax, codebook=CODEBOOK):
    codes = unpack_nibbles(packed, n)
    out = [float(codebook[int(c)]) * float(absmax) for c in codes]
    return np.array(out, dtype=np.float64)


_rng = np.random.RandomState(1234)

CASES = [
    [],
    [0],
    [15],
    [3, 10],
    [10, 3],
    [3, 10, 1, 8, 5],
    list(range(16)),
    list(range(15, -1, -1)),
    [(i * 7 + 3) % 16 for i in range(17)],
]
for _n in (5, 8, 9, 32, 33, 64, 65, 100):
    CASES.append(_rng.randint(0, 16, size=_n).tolist())

_ABSMAXES = [0.0, 1.0, 0.003, 127.5, 3.14159, 42.0, 255.0]

CONFIGS = [([], 5.0)]
for _codes in CASES:
    if len(_codes) == 0:
        continue
    for _absmax in _ABSMAXES:
        CONFIGS.append((_codes, _absmax))
