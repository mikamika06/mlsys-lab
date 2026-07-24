## Context

FP8 E4M3 (as used in ML inference/training, e.g. `torch.float8_e4m3fn`) packs
a value into 8 bits: 1 sign bit, 4 exponent bits (bias $7$), 3 mantissa bits.

For a stored exponent field $e$ and mantissa field $m$:

$$
\text{value} =
\begin{cases}
(-1)^s \left(1 + \dfrac{m}{8}\right) 2^{\,e-7}, & e \ge 1 \quad\text{(normal)} \\[4pt]
(-1)^s \dfrac{m}{8}\, 2^{-6}, & e = 0 \quad\text{(subnormal)}
\end{cases}
$$

The largest finite magnitude is $448$ (at $e=14, m=7$); the code
$e=15, m=7$ is reserved and never produced by encoding. Any value whose
magnitude exceeds $448$ must **saturate** to $\pm 448$ rather than
overflow to infinity — E4M3 has no infinities.

Encoding uses **round-to-nearest-even (RNE)**: a value is mapped to
whichever of its two neighboring representable grid points is closer; on
an exact tie, the neighbor whose stored mantissa bit pattern has an even
low bit is chosen.

## Task

Implement `e4m3_round_trip(x)`:

```python
def e4m3_round_trip(x: np.ndarray) -> np.ndarray:
    ...
```

Given a NumPy array `x` of arbitrary float values, simulate encoding each
value to E4M3 and decoding it straight back — i.e. return the array of
values E4M3 would actually be able to represent after round-tripping `x`
through it:

1. Clamp `|x|` to `448` (saturation) before rounding.
2. Round to the nearest representable E4M3 magnitude, using RNE on exact
   ties (this affects both the subnormal range near zero and every normal
   exponent block).
3. Restore the original sign, including the sign of zero (`x == 0` should
   round-trip to a zero of the same sign).

Return a `float32` array of the same shape as `x`. No lookup table of
"expected outputs" is meaningful here — round by the actual bit-field
formulas above (e.g. build the 128 nonnegative representable magnitudes
from `e`/`m` directly and search against them), not a hardcoded list of
example results.

## Example

```python
import numpy as np

x = np.array([0.0, 1.0, 0.5, 300.0, 448.0, 1000.0, -1000.0])
e4m3_round_trip(x)
# array([   0. ,    1. ,    0.5,  288. ,  448. ,  448. , -448. ], dtype=float32)
# (300 rounds to the nearest E4M3 grid point, 288; 1000 and -1000 saturate)
```

## What the gate checks

The gate builds a fixed (non-random) probe array from the real E4M3 grid:
values that already sit exactly on a grid point, values placed exactly
halfway between two adjacent grid points — both in the subnormal region
near zero and in several different normal exponent blocks, to exercise
round-to-even in both directions — ordinary off-grid values needing plain
nearest-neighbor rounding, values at and past the $\pm448$ saturation
boundary, and $\pm0.0$.

Your output is compared element-for-element against the oracle's
(`max_abs_err`, threshold `0.0`, i.e. bit-exact) — the oracle computes its
expected values fresh from the same bit-field formulas each run, nothing
is hardcoded. Missing the subnormal branch, breaking ties toward-nearest
instead of to-even, forgetting to saturate before rounding, or losing the
sign of zero will produce a nonzero error on at least one probe value.
