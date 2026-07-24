## Context

GPTQ quantization improves over independent rounding by compensating later columns after quantizing each column. A simplified GPTQ step maintains a working weight matrix $W$ and an inverse Hessian matrix $H^{-1}$.

For column $j$, GPTQ first quantizes the current column:

$$
q_j = Q(W_j)
$$

The quantization error is then propagated into the remaining columns:

$$
W_k \leftarrow W_k + (q_j - W_j)\frac{(H^{-1})_{j,k}}{(H^{-1})_{j,j}},
\qquad k > j .
$$

This update is the part that distinguishes GPTQ from round-to-nearest (RTN). If the remaining-column update is skipped, every column is quantized independently and the method degenerates to RTN.

The simplified quantizer uses symmetric per-column scaling. For a column $w$, with $b$ bits, the quantization range is

$$
[-(2^{b-1}-1), 2^{b-1}-1],
$$

and the scale is

$$
s = \frac{\max(|w|)}{2^{b-1}-1}.
$$

The quantized values are reconstructed as

$$
Q(w) = s \cdot \mathrm{clip}\left(\mathrm{round}(w/s), -(2^{b-1}-1), 2^{b-1}-1\right).
$$

## Task

Implement `gptq_quantize(W, H_inv, bits=4)`.

The function receives:

```python
def gptq_quantize(W: np.ndarray, H_inv: np.ndarray, bits: int = 4) -> np.ndarray:
    ...
```

where `W` is a floating point matrix of shape $(m, n)$ and `H_inv` is an $n \times n$ inverse Hessian matrix. Return a floating point matrix containing the GPTQ-quantized weights.

Process columns from left to right. After quantizing column $j$, update all remaining columns using the corresponding entries from `H_inv`. Do not replace `H_inv` with the Hessian matrix itself.

## Example

```python
import numpy as np

W = np.array([
    [1.1, -0.7, 0.4],
    [0.3,  1.2, -0.8],
], dtype=np.float64)

H_inv = np.array([
    [1.0, 0.2, 0.1],
    [0.2, 1.0, 0.3],
    [0.1, 0.3, 1.0],
], dtype=np.float64)

Q = gptq_quantize(W, H_inv, bits=4)
```

The output is the compensated GPTQ result, not the result of quantizing each column independently.

## What the gate checks

The gate computes the GPTQ reference implementation directly from the algorithm above using NumPy operations. The returned matrix must match the oracle with relative error

$$
\frac{\lVert Q_{\mathrm{student}} - Q_{\mathrm{ref}}\rVert}
{\lVert Q_{\mathrm{ref}}\rVert + 10^{-12}}
\le 10^{-9}.
$$

The gate also computes the independent RTN result and verifies that the submitted implementation is different from the degenerate version that skips the remaining-column update.
