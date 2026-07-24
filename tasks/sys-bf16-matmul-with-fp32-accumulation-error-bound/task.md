## Context

Bfloat16 (BF16) keeps the sign bit and exponent range of float32 while reducing the
number of fraction bits. A common mixed-precision strategy stores inputs in BF16
but performs accumulation in float32.

For matrices $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$,
the reference result is

$$
C_{ij} = \sum_{r=1}^{k} A_{ir}B_{rj}.
$$

The mixed-precision computation first rounds inputs to BF16:

$$
\hat{A} = \operatorname{bf16}(A), \qquad \hat{B} = \operatorname{bf16}(B),
$$

then accumulates products in float32:

$$
\hat{C}_{ij} = \operatorname{fp32}\left(
\sum_{r=1}^{k} \hat{A}_{ir}\hat{B}_{rj}
\right).
$$

Using BF16 for the accumulation itself introduces additional rounding after each
addition and usually increases error. The task is to preserve the FP32 accumulation
behavior.

## Task

Implement `bf16_matmul_fp32_accum(A, B)`:

```python
def bf16_matmul_fp32_accum(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ...
```

The inputs are 2-D NumPy arrays containing float values with compatible matrix
multiplication shapes. The function must:

1. Round both inputs to BF16 precision.
2. Multiply the BF16 values.
3. Accumulate the dot products using float32 accumulation.
4. Return a float32 NumPy array.

Do not use float64 matrix multiplication as the implementation strategy.

## Example

```python
import numpy as np

A = np.array([[1.1, 2.2]], dtype=np.float32)
B = np.array([[3.3], [4.4]], dtype=np.float32)

C = bf16_matmul_fp32_accum(A, B)
# C contains the BF16-input, FP32-accumulated result
```

## What the gate checks

The gate computes the FP64 NumPy matrix multiplication as the numerical oracle and
measures the relative error

$$
\mathrm{rel\_err} =
\frac{\lVert C_{\mathrm{candidate}} - C_{\mathrm{fp64}}\rVert_2}
{\lVert C_{\mathrm{fp64}}\rVert_2 + 10^{-12}}.
$$

The result must stay below the allowed mixed-precision error bound. The checker
also verifies that a BF16-input implementation with BF16 accumulation has larger
error on the same cases, so the task distinguishes FP32 accumulation from lower
precision accumulation.
