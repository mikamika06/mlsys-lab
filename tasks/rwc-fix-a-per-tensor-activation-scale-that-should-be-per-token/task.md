## Context

Production inference systems often quantize activations to reduce memory bandwidth and improve throughput. A common symmetric int8 activation quantization scheme represents a tensor $X$ using an integer tensor $Q$ and a scale factor $s$:

$$
Q = \operatorname{round}(X / s), \qquad \hat{X} = Qs .
$$

A per-tensor scale uses one value for the entire activation batch:

$$
s = \frac{\max_{i,j}|X_{ij}|}{127}.
$$

This can fail when one token contains an extreme outlier. The outlier determines the scale, causing values from other tokens to be rounded into fewer integer levels.

For a token batch $X \in \mathbb{R}^{n \times d}$, a per-token scale computes one scale for each row:

$$
s_i = \frac{\max_j |X_{ij}|}{127}.
$$

The quantized output is reconstructed as

$$
\hat{X}_{ij} = \operatorname{round}(X_{ij}/s_i)s_i .
$$

This preserves more precision because each token uses the full int8 range independently.

## Task

Implement `per_token_int8_dequant(X)`.

The function receives a 2-D NumPy array `X` of shape $(n, d)$ containing activation values and returns a float64 NumPy array with the same shape. The function must apply symmetric int8 quantization and dequantization using a separate scale for every token (row).

Use the formula:

```python
scale = max(abs(X[row])) / 127
```

for each row. If a row has scale zero, return zeros for that row instead of dividing by zero.

Do not use a single scale for the complete matrix.

## Example

```python
import numpy as np

X = np.array([
    [100.0, -80.0, 20.0],
    [0.5, -0.25, 0.1],
])

Y = per_token_int8_dequant(X)

# The first row uses scale 100/127.
# The second row uses scale 0.5/127.
```

## What the gate checks

The gate creates outlier-heavy activation batches and computes the reference result using the per-token quantization algorithm. The returned array is compared with the NumPy oracle using maximum absolute error:

$$
\max_{i,j} |\hat{X}_{ij}^{student} - \hat{X}_{ij}^{reference}|.
$$

A solution that uses one per-tensor scale has a larger reconstruction error on these cases and does not pass.
