## Context

FlashAttention computes attention without materializing the full attention matrix. For query matrix $Q$, key matrix $K$, and value matrix $V$, the forward pass is

$$
S = \frac{QK^\top}{\sqrt{d}}, \qquad
P = \mathrm{softmax}(S), \qquad
O = PV .
$$

The backward pass recomputes the attention probabilities and applies the chain rule. Given an upstream gradient $dO$, the gradients are

$$
dV = P^\top dO ,
$$

$$
dP = dO V^\top ,
$$

$$
dS_i = P_i \left(dP_i - \sum_j dP_{ij}P_{ij}\right),
$$

$$
dQ = \frac{dS K}{\sqrt{d}}, \qquad
dK = \frac{dS^\top Q}{\sqrt{d}} .
$$

Modern accelerators often use reduced precision. This task emulates bfloat16 by rounding intermediate tensors after each major operation. The backward kernel must recompute probabilities using the same rounding behavior.

## Task

Implement `flash_bwd_bf16(Q, K, V, dO)`.

The inputs are NumPy arrays with shape $(n, d)$ for `Q`, `K`, `V`, and `dO`. Return a tuple `(dQ, dK, dV)` containing the gradients of the attention operation.

Requirements:

- Use NumPy operations only.
- Emulate bfloat16 by rounding intermediate values to bfloat16 precision.
- Accumulate `dQ` serially over key/value tiles during the backward computation.
- Return `float32` arrays.
- The implementation should match the bf16-emulated reference backward pass.

## Example

```python
import numpy as np

Q = np.array([[1, 0], [0, 1]], dtype=np.float32)
K = np.array([[1, 0], [0, 1]], dtype=np.float32)
V = np.array([[2, 3], [4, 5]], dtype=np.float32)
dO = np.ones((2, 2), dtype=np.float32)

dQ, dK, dV = flash_bwd_bf16(Q, K, V, dO)
```

## What the gate checks

The gate builds small attention problems and computes the expected gradients with an independent NumPy bf16-emulated reference implementation. It compares every returned gradient tensor with the oracle using

$$
\max_{i} |x_i - \hat{x}_i| .
$$

The maximum absolute error must be below $5 \times 10^{-3}$. Implementations that omit bf16 rounding or use an incorrect backward formula fail the check.
