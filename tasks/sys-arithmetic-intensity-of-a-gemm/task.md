## Context

The *arithmetic intensity* of a computation is the ratio of floating‑point operations (FLOPs) to memory traffic measured in bytes. For a matrix multiplication (GEMM)

$$C = A \times B,$$

with $A\in\mathbb{R}^{M\times K}$, $B\in\mathbb{R}^{K\times N}$ and $C\in\mathbb{R}^{M\times N}$, the number of FLOPs is

$$\text{FLOPs}=2\,M\,K\,N,$$

because each output element requires $K$ multiply‑add pairs.  
The memory traffic consists of reading all elements of $A$ and $B$, and writing all elements of $C$. If each element occupies $\text{bytes}$ bytes, then

$$\text{Bytes}= (M K + K N + M N)\times \text{bytes}.$$

Thus the arithmetic intensity is

$$I = \frac{\text{FLOPs}}{\text{Bytes}}.$$

This metric appears in the *roofline model*, which bounds achievable performance on a given hardware platform.

## Task

Implement `arithmetic_intensity(m, k, n, dtype)` that returns the arithmetic intensity of a GEMM with shapes $(m,k)$ and $(k,n)$ using the provided NumPy data type `dtype`. The function should:

1. Allocate zero arrays of the appropriate shape and dtype.
2. Compute the total bytes read from $A$ and $B$, plus the bytes written to $C$.
3. Return the ratio $\frac{2\,m\,k\,n}{\text{total bytes}}$ as a Python `float`.

The implementation must use NumPy only; no explicit loops or other libraries.

## Example

```python
import numpy as np
I = arithmetic_intensity(10, 20, 30, np.float32)
# I ≈ 2*10*20*30 / ((10*20 + 20*30 + 10*30) * 4)
```

## What the gate checks

The grader computes a reference intensity using NumPy’s `array.nbytes` to obtain the exact memory footprint. It then compares your result against this oracle, requiring a relative error $\le 10^{-9}$.
