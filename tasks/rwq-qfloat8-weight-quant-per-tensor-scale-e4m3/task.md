## Context

FP8 weight quantization ("qfloat8") composes two steps: a single
**per-tensor FP32 scale** brings the weight tensor's range onto FP8's
representable range, then every scaled element is **cast** onto the
FP8 grid (rather than rounded onto a uniform integer grid the way
int8 quantization is).

Here the FP8 format is **E4M3** (1 sign, 4 exponent, 3 mantissa; bias
7; no infinities; a single NaN code `S.1111.111`), whose largest finite
magnitude is $448$:

$$
\text{scale} = \frac{\max(|W|)}{448}, \qquad
W' = \frac{W}{\text{scale}}
$$

$$
\hat W'_i = \mathrm{round\_to\_e4m3}(W'_i), \qquad
\hat W_i = \hat W'_i \cdot \text{scale}
$$

where `round_to_e4m3` snaps to the *nearest* representable E4M3 value
(magnitudes $\left(1+\tfrac{m}{8}\right)\cdot2^{e-7}$ for normal
exponents $e\in\{1,\dots,14\}$, subnormals $\tfrac{m}{8}\cdot2^{-6}$ at
$e=0$, and $e=15$ finite for $m\in\{0,\dots,6\}$ — unlike int8, the
grid is *non-uniform*: spacing is fine near 0 and coarse near 448).

## Task

Implement `qfloat8_weight_quant`:

```python
def qfloat8_weight_quant(W: np.ndarray):
    ...
```

- `W`: any-shape `float64` array (may contain both signs and zero).

1. Compute `scale = max(|W|) / 448.0` (if `W` is all-zero, use
   `scale = 1.0`).
2. Divide: `W_scaled = W / scale`.
3. Cast every element of `W_scaled` to the **nearest** representable
   signed E4M3 value — build the full grid by enumerating all 256
   `sign × exponent × mantissa` combinations (excluding the two NaN
   codes) and, for each element, pick whichever grid value minimizes
   absolute difference.
4. Dequantize: `W_hat = e4m3_values * scale`.

Return `(scale, e4m3_values, W_hat)`:
- `scale`: Python `float`.
- `e4m3_values`: `float64` array, same shape as `W`, the *values* each
  element snapped to on the E4M3 grid (not bit patterns).
- `W_hat`: `float64` array, same shape as `W`, the final dequantized
  reconstruction (`e4m3_values * scale`).

## Example

```python
import numpy as np
W = np.array([100.0, -0.3, 0.0, 50.0])
scale, codes, W_hat = qfloat8_weight_quant(W)
# scale = 100.0 / 448.0
# codes[0] is whichever E4M3 grid value is closest to 100.0/scale == 448.0
#   (i.e. exactly 448.0 -- the loudest element always lands on the grid's
#   max magnitude by construction of the scale)
# W_hat[0] == codes[0] * scale  ~=  100.0
```

## What the gate checks

The grader builds several seeded weight tensors (mixed sign, mixed
magnitude) and computes the reference scale/grid-cast/dequant
independently in NumPy, using the exact algorithm above (same E4M3
grid, same 448 normalizer).

`scale_abs_err` is the worst-case absolute difference between your
`scale` and the oracle's, across all cases (must be `<= 1e-9`) — this
is exact floating-point arithmetic, so it isolates a wrong normalizer
(e.g. using 240 like E5M2, or forgetting to divide by `max(|W|)`).
`dequant_max_abs_err` is the worst-case max elementwise absolute
difference between your `W_hat` and the oracle's (must be `<= 1e-4`,
matching E4M3's coarsest grid spacing near 448) — this catches a wrong
or incomplete E4M3 grid (e.g. missing subnormals, or clamping instead
of rounding to nearest) even when the scale itself is correct.
