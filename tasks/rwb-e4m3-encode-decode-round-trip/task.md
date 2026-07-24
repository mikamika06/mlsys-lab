## Context

**E4M3FN** (1 sign bit, 4 exponent bits, 3 mantissa bits, bias 7, no
infinities, saturating) is the 8-bit float format used for weight/activation
storage in FP8 training and inference. For an 8-bit code with sign $s$,
exponent field $e$, mantissa field $m$ (each read as unsigned integers):

$$
\mathrm{value}(s,e,m) =
\begin{cases}
(-1)^s \cdot \dfrac{m}{8} \cdot 2^{-6} & e = 0 \ \ (\text{subnormal}) \\[4pt]
(-1)^s \cdot \left(1 + \dfrac{m}{8}\right) \cdot 2^{\,e-7} & 1 \le e \le 14 \ \ (\text{normal}) \\[4pt]
\mathrm{NaN} & e = 15,\ m = 7 \ \ (\text{the one reserved code, either sign})
\end{cases}
$$

There is no infinity — every other $e=15$ pattern is an ordinary finite
value, and the largest representable magnitude is $448$ (code
`0111_0110`, i.e. $e=14, m=6$: $(1+6/8)\cdot 2^{7}=448$). Casting a value
whose magnitude exceeds $448$ **saturates** to $\pm 448$ rather than
overflowing to infinity or NaN. Casting a value that isn't exactly
representable rounds to the nearest code, ties resolved **to even**
(round-to-nearest-even, RNE).

## Task

Implement both directions:

```python
def encode_e4m3(x: np.ndarray) -> np.ndarray:
    ...

def decode_e4m3(codes: np.ndarray) -> np.ndarray:
    ...
```

- `encode_e4m3(x)` — `x` is a `float32`/`float64` array. Saturate to
  $\pm 448$, then round-to-nearest-even to the closest representable E4M3FN
  value. Return a `uint8` array of the same shape holding the packed
  `sign(1) | exponent(4) | mantissa(3)` codes ($\pm 0$ must be preserved:
  the sign bit of an input `-0.0` must be set).
- `decode_e4m3(codes)` — `codes` is a `uint8` array of raw E4M3FN byte
  patterns (every value $0$–$255$ is a valid input, including the two NaN
  codes). Return a `float32`/`float64` array of the decoded values, per the
  formula above.

## Example

```python
import numpy as np

x = np.array([0.0, 1.0, 100.0, 500.0, -0.3])
codes = encode_e4m3(x)
# 500.0 saturates to 448.0; the rest round to the nearest E4M3FN grid point
back = decode_e4m3(codes)
print(back)  # [0.0, 1.0, 96.0 or 104.0 (nearest grid point), 448.0, -0.3125 (nearest grid point)]
```

## What the gate checks

Two independent, exact checks feed one `max_abs_err`:

1. **decode** — `decode_e4m3` is called on all 256 possible byte values and
   compared to the real bit-pattern decode formula above (NaN codes must
   decode to NaN; every other code must match exactly).
2. **encode** — `encode_e4m3` is called on a fixture of probe values
   (`x.npy`: exact grid points, RNE tie midpoints, saturating values,
   subnormal-range magnitudes, and broad random coverage). Your returned
   codes are then decoded with the *same* verified bit-pattern formula and
   compared to the reference saturate-then-round-to-nearest-even value.

Both must match **exactly** — `max_abs_err <= 0.0`. Any rounding-mode
mismatch (e.g. round-half-up instead of round-half-to-even), a missed
saturation clamp, or a sign bit dropped on `-0.0` will show up as a nonzero
error on the corresponding probe value.
