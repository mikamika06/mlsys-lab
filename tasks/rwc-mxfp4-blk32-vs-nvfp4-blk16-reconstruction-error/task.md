## Context

Both MXFP4 and NVFP4 store each weight as a 4-bit E2M1 float (1 sign
bit, 2 exponent bits, 1 mantissa bit), whose nonnegative magnitudes form
the grid

$$
\{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\},
$$

shared by a block of elements through one scale. They differ in **block
size** and in how precisely the shared scale itself can be represented:

- **MXFP4**: blocks of 32 elements; the scale is an **E8M0** value —
  exponent-only, no mantissa — so it can only be a power of two. For a
  block with $a = \max(|\text{block}|)$, the largest safe power-of-two
  scale is
  $$
  s_{\text{MX}} = 2^{\left\lceil \log_2(a / 6) \right\rceil} .
  $$
- **NVFP4**: blocks of just 16 elements; the scale is computed as
  $a/6$ and then itself rounded to the nearest real **FP8 E4M3** value
  (3 mantissa bits) — far less coarse than a bare power of two.

Smaller blocks mean less magnitude variation for one shared scale to
cover, and a non-power-of-two scale wastes less of the FP4 grid's
resolution — both push NVFP4 toward lower reconstruction error, at the
cost of one extra scale byte for every 16 (not 32) elements.

## Task

Implement `compare_mxfp4_nvfp4`:

```python
def compare_mxfp4_nvfp4(weights: np.ndarray) -> np.ndarray:
    ...
```

For the given `weights` (any shape), quantize-then-dequantize with both
schemes:

1. **MXFP4** (block 32): for each block, $a = \max(|\text{block}|)$; if
   $a = 0$ the scale is 1, otherwise $s = 2^{\lceil \log_2(a/6)\rceil}$.
   Divide the block by $s$, clip magnitudes to $[0, 6]$, snap each to
   the nearest FP4 grid value (preserving sign), then multiply back by
   $s$.
2. **NVFP4** (block 16): same per-element grid and clip-and-snap
   procedure, but the scale is $a/6$ rounded to the nearest real FP8
   E4M3 value (round-to-nearest-even against the true E4M3 grid,
   saturating at its max representable magnitude) instead of being
   restricted to a power of two.

Return `np.array([mxfp4_rel_err, nvfp4_rel_err])`, where each entry is
the global relative L2 reconstruction error
$\lVert \hat w - w \rVert / \lVert w \rVert$ of that scheme's
dequantized output against the original `weights`.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
w = rng.standard_normal(1000)

errs = compare_mxfp4_nvfp4(w)
# errs[1] (NVFP4) is consistently lower than errs[0] (MXFP4) for
# generic (non-power-of-two-friendly) weight distributions.
```

## What the gate checks

The gate, **rel_err**, compares your 2-element array against an fp64
NumPy oracle across several distributions (standard normal, a
heavy-tailed Student-t, uniform, and a mostly-zero block edge case).
Your result must match the oracle to a relative error `<= 1e-9`; using
the wrong block size, restricting NVFP4's scale to a power of two, or
swapping which scheme gets which block size all produce a different,
failing result.
