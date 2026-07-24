## Context

NVFP4 stores a tensor with **two levels of scaling** on top of 4-bit
elements. Nothing in storage is a plain float except one scalar:

- one tensor-level scale $g$, stored as a plain `float32`;
- one **E4M3** byte per block of 16 elements, encoding that block's
  positive scale $s_b$ (1 sign bit, 4 exponent bits, 3 mantissa bits,
  bias 7 — the sign bit is always `0` here since a scale is never
  negative);
- one **E2M1** nibble per element, encoding its code $q_i$ (1 sign bit,
  2 exponent bits, 1 mantissa bit, bias 1).

**Decoding E4M3** ($S$=sign bit, $E$=4-bit exponent, $M$=3-bit mantissa):

$$
s = (-1)^S \times
\begin{cases}
2^{\,E-7}\left(1+\dfrac{M}{8}\right) & E \neq 0 \\[6pt]
2^{-6}\,\dfrac{M}{8} & E = 0
\end{cases}
$$

**Decoding E2M1** ($S$=sign bit, $E$=2-bit exponent, $M$=1-bit mantissa):

$$
q = (-1)^S \times
\begin{cases}
2^{\,E-1}\left(1+\dfrac{M}{2}\right) & E \neq 0 \\[6pt]
\dfrac{M}{2} & E = 0
\end{cases}
$$

(this reproduces the familiar 8-level magnitude grid
$\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$, mirrored by the sign bit).

**Reconstruction.** For element $i$ inside block $b$:

$$
\hat x_i = g \cdot s_b \cdot q_i .
$$

## Task

Implement `nvfp4_reconstruct(global_scale, e4m3_block_codes, e2m1_codes)`:

```python
import numpy as np

def nvfp4_reconstruct(global_scale: float, e4m3_block_codes: np.ndarray, e2m1_codes: np.ndarray) -> np.ndarray:
    ...
```

- `global_scale`: a Python float (or 0-d array), the tensor-level $g$ —
  already plain `float32`, no decoding needed.
- `e4m3_block_codes`: `uint8` array of shape `(n_blocks,)`, one raw E4M3
  byte per block.
- `e2m1_codes`: `uint8` array of shape `(n_blocks, 16)`, one raw E2M1
  nibble (`0..15`) per element, 16 elements per block.

Decode both code arrays with the formulas above, then combine to return a
`(n_blocks, 16)` `float64` array of reconstructed values
$\hat x_{b,i} = g \cdot s_b \cdot q_{b,i}$ (broadcast each block's decoded
scale $s_b$ across its 16 elements).

## Example

```python
import numpy as np
g = 2.0
e4m3_block_codes = np.array([0b0_0111_000], dtype=np.uint8)  # S=0,E=7,M=0 -> s=2**0*(1+0)=1.0
e2m1_codes = np.array([[0b1_01_1]], dtype=np.uint8)           # S=1,E=1,M=1 -> q=-(2**0*1.5)=-1.5
nvfp4_reconstruct(g, e4m3_block_codes, e2m1_codes)
# array([[-3.0]])   (2.0 * 1.0 * -1.5)
```

## What the gate checks

The grader loads three committed fixtures (`e4m3_block_codes.npy`,
`e2m1_codes.npy`, `global_scale.npy` — 40 blocks of 16 elements, block
scale codes spanning the full non-NaN E4M3 range, element codes covering
all 16 E2M1 signed values), decodes both code arrays with an independent
NumPy oracle using the exact bit-field formulas above, and reconstructs
$\hat x = g \cdot s_b \cdot q_i$.

The gate metric is `max_abs_err`, the largest element-wise absolute
difference between your reconstruction and the oracle's; it must be
`< 1e-6`. A wrong E4M3 or E2M1 bias/subnormal branch, a broadcasting bug
that misaligns block scales with their 16 elements, or omitting the
global scale will all produce a mismatch that exceeds the threshold.
