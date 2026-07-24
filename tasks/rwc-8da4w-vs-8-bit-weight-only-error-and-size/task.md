## Context

A linear layer computes

$$Y = XW^\top$$

where $X \in \mathbb{R}^{m \times d}$ contains activations and
$W \in \mathbb{R}^{n \times d}$ contains weights.

Weight-only 8-bit quantization stores integer weights and reconstructs them using
a scale per output channel:

$$\hat{W}_{i,j}=s_i q_{i,j}, \qquad q_{i,j}\in[-127,127].$$

The reconstructed output is

$$\hat{Y}=X\hat{W}^{\top}.$$

The 8da4w approach quantizes activations to int8 and weights to signed int4:

$$\hat{Y}=(s_x X_q)(s_w W_q)^\top,$$

where $X_q$ uses the range $[-127,127]$ and $W_q$ uses the range $[-7,7]$.

The two approaches trade off accuracy and storage. For a layer with $n$ output
channels and $d$ input features, 8-bit weight-only storage requires

$$S_{wo8}=nd+4n$$

bytes because each weight uses one byte and each channel stores a float32 scale.

The packed int4 weights in 8da4w require

$$S_{8da4w}=\left\lceil\frac{nd}{2}\right\rceil+4n$$

bytes.

## Task

Implement `compare_linear_quantization(W, X)`:

```python
def compare_linear_quantization(W: np.ndarray, X: np.ndarray) -> dict:
    ...
```

`W` is a float32 matrix of shape $(n,d)$ and `X` is a float32 matrix of shape
$(m,d)$.

Return a dictionary with exactly these keys:

- `"error_8da4w"`: relative L2 error between $XW^\top$ and the reconstructed 8da4w output.
- `"error_wo8"`: relative L2 error between $XW^\top$ and the reconstructed 8-bit weight-only output.
- `"size_8da4w"`: original weight bytes divided by packed int4 storage bytes.
- `"size_wo8"`: original weight bytes divided by int8 weight-only storage bytes.
- `"tradeoff"`: `1.0` when 8da4w has smaller storage and no larger error than 8-bit weight-only, otherwise `0.0`.

Use per-output-channel scales for weights. Use NumPy operations for the quantization and reconstruction.

## Example

```python
import numpy as np

W = np.array([[1.0, -2.0], [0.5, 1.5]], dtype=np.float32)
X = np.array([[1.0, 2.0]], dtype=np.float32)

result = compare_linear_quantization(W, X)
```

## What the gate checks

The grader computes both quantization schemes independently with a NumPy oracle.
It compares the returned errors, size ratios, and tradeoff decision against the
oracle values.

Returning fixed constants or using a different quantization rule fails because
the test cases include different layer shapes and value distributions.
