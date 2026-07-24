"""Deterministic float64 test vector for num-exact-rational-value-of-each-float.

Mixes hand-picked corner cases (zeros, subnormals, min/max normals, powers of two)
with seeded random values whose exponents are scattered across the whole range.

    python3 tasks/num-exact-rational-value-of-each-float/gen_fixtures.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "fixtures"

SPECIALS = [
    0.0, -0.0, 1.0, -1.0, 0.5, -2.0, 0.1, -0.1, 1.0 / 3.0, 2.0 / 3.0,
    3.141592653589793, 2.718281828459045,
    5e-324,                       # smallest positive subnormal
    1e-310,                       # subnormal
    -3e-320,                      # negative subnormal
    2.2250738585072014e-308,      # smallest positive normal
    1.7976931348623157e308,       # largest finite
    float(2 ** 52 + 1),
    float(2 ** 53),
    1e16 + 2.0,
    123456789.123456789,
    -1e-5,
    9007199254740993.0,
]


def build():
    rng = np.random.default_rng(0)
    vals = list(SPECIALS)
    mant = rng.uniform(0.5, 1.0, size=48)
    expo = rng.integers(-1060, 1000, size=48)
    sign = rng.choice([-1.0, 1.0], size=48)
    for m, e, s in zip(mant, expo, sign):
        v = s * np.ldexp(float(m), int(e))
        if np.isfinite(v):
            vals.append(float(v))
    return np.array(vals, dtype=np.float64)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    x = build()
    np.save(OUT / "values.npy", x)
    print("wrote", x.shape, "all finite:", bool(np.all(np.isfinite(x))))
