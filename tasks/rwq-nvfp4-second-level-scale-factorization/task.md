## Context

NVFP4 stores each weight element in E2M1 — a 4-bit float (1 sign, 2
exponent, 1 mantissa) whose largest representable magnitude is exactly
$6.0$. A 4-bit grid that small can only ever cover values up to 6, so
NVFP4 uses **two levels of scaling**: one scalar FP32 `per_tensor_scale`
for the whole tensor, and one **per-block** scale (block size 16)
stored in E4M3 (8-bit float: 1 sign, 4 exponent, 3 mantissa, no
infinities, one NaN code `S.1111.111`, max finite magnitude $448$).

For a block $g$ of 16 consecutive weights, the block scale is chosen so
that the block's largest element lands exactly on E2M1's largest
representable value ($6.0$) once both scale levels are divided out:

$$
\text{raw\_scale}(g) = \frac{\max(|g|)}{6 \cdot \text{per\_tensor\_scale}}
$$

That raw scale is itself only representable to E4M3 precision — it
gets **cast to the nearest E4M3 grid value** before being stored
alongside the block:

$$
\text{block\_scale}(g) = \mathrm{round\_to\_e4m3}\big(\text{raw\_scale}(g)\big)
$$

(E4M3 magnitudes: normals $\left(1+\tfrac{m}{8}\right)\cdot 2^{e-7}$ for
exponent field $e \in \{1,\dots,14\}$, mantissa $m\in\{0,\dots,7\}$;
subnormals $\tfrac{m}{8}\cdot 2^{-6}$ for $e=0$; and $e=15$ is finite
for $m\in\{0,\dots,6\}$ — only $m=7$ at $e=15$ is NaN.)

## Task

Implement `nvfp4_block_scales`:

```python
def nvfp4_block_scales(W: np.ndarray, group_size: int, per_tensor_scale: float) -> np.ndarray:
    ...
```

- `W`: 1-D `float64` array, `len(W)` a multiple of `group_size`.
- `group_size`: block size (NVFP4 uses 16).
- `per_tensor_scale`: positive Python `float`, the tensor's single FP32 scale.

For each contiguous block of `group_size` elements of `W`:

1. Compute `raw_scale = max(|block|) / (6.0 * per_tensor_scale)`.
2. Round `raw_scale` to the **nearest** representable (non-negative)
   E4M3 magnitude — build the full E4M3 grid (enumerate all
   `exponent × mantissa` combinations as above, excluding the NaN
   code) and pick whichever grid value minimizes
   `abs(grid_value - raw_scale)`.

Return the array of per-block E4M3-quantized scales, shape
`(len(W) // group_size,)`, `float64`.

## Example

```python
import numpy as np
W = np.zeros(16)
W[3] = 12.0          # block absmax = 12.0
per_tensor_scale = 1.0
scales = nvfp4_block_scales(W, group_size=16, per_tensor_scale=per_tensor_scale)
# raw_scale = 12.0 / (6.0 * 1.0) = 2.0, which happens to sit exactly
# on the E4M3 grid ((1 + 0/8) * 2**1 == 2.0), so scales[0] == 2.0.
```

## What the gate checks

The grader builds several seeded `W` / `group_size` / `per_tensor_scale`
cases spanning a wide range of block magnitudes and computes the
reference block scales independently in NumPy: the exact same
`max(|block|) / (6 * per_tensor_scale)` formula, then nearest-neighbor
rounding onto an E4M3 grid built by enumerating all 256 8-bit patterns
(sign, 4-bit exponent, 3-bit mantissa) and excluding the single NaN
code.

`rel_err` is the global relative L2 error between your returned
per-block scales (concatenated across all cases) and the oracle's
(must be `<= 1e-4`). Getting the identity right but skipping the E4M3
rounding step (returning the raw float scale) or using the wrong E4M3
grid (e.g. treating the format as symmetric like INT8, or missing the
subnormal range) both produce a visible, non-negligible error here.
