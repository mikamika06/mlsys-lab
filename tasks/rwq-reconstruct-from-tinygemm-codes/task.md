## Context

TinyGEMM (and similar low-bit GEMM kernels) store 4-bit quantized weights as
`uint4` integer codes packed into byte arrays. To reconstruct the original
floating-point values the kernel applies a simple **float-domain dequantization**
formula per code:

$$\hat{w} = q \cdot s + z$$

where $q \in \{0, 1, \ldots, 15\}$ is the unsigned 4-bit code, $s$ is the
per-group (or per-tensor) floating-point **scale**, and $z$ is the floating-point
**zero-point** (stored in float, NOT the integer zero-point used in some other
conventions).

This is the *float zero-point* convention: both $s$ and $z$ are floats and the
reconstruction is purely additive — no subtraction of an integer zero-point
before scaling. The formula exactly matches what `bitsandbytes` and the original
TinyGEMM reference use when they store the zero-point in float32.

## Task

Implement `dequantize_uint4(codes, scale, zero_point)`:

```python
def dequantize_uint4(codes, scale, zero_point):
    ...
```

- `codes`: integer NumPy array of shape `(n,)` with values in `[0, 15]`
  (already unpacked — no bit-unpacking needed).
- `scale`: scalar float, the group scale $s$.
- `zero_point`: scalar float, the float-domain zero-point $z$.

Return a `float32` NumPy array of shape `(n,)` containing
$\hat{w}_i = \text{codes}_i \cdot s + z$.

## Example

```python
import numpy as np
codes = np.array([0, 7, 8, 15], dtype=np.uint8)
scale = 0.1
zero_point = -0.75
out = dequantize_uint4(codes, scale, zero_point)
# out ≈ [-0.75, -0.05,  0.05,  0.75]
```

Derivation: $0 \cdot 0.1 - 0.75 = -0.75$, $7 \cdot 0.1 - 0.75 = -0.05$, etc.

## What the gate checks

The grader generates random `uint4` codes, scales, and zero-points, applies the
reference formula $\hat{w} = q \cdot s + z$, and compares against your output
using `max_abs_err`. The gate passes when the maximum absolute error is
$\le 10^{-6}$.
