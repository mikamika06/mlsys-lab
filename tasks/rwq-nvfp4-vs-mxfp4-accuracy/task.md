## Context

Low-bit floating point formats reduce memory bandwidth and storage cost by
representing values with small floating point codes and shared scaling factors.

This task compares two block scaling strategies. NVFP4 uses a scale value for
each block of 16 elements, while MXFP4 uses a power-of-two scale for each block
of 32 elements.

For a block of values $x$, quantization reconstructs an approximation
$\hat{x}$ using a scale $s$ and a small floating point code:

$$
\hat{x_i} = s \cdot q_i .
$$

The reconstruction error is measured with root mean squared error:

$$
\mathrm{RMSE}(x,\hat{x}) =
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x_i})^2}.
$$

Smaller scaling blocks can adapt more closely to local magnitudes, which often
reduces reconstruction error.

## Task

Implement `fp4_accuracy_comparison(weight)`:

```python
def fp4_accuracy_comparison(weight: np.ndarray) -> tuple[float, float]:
    ...
```

The function receives a one-dimensional NumPy array of `float32` weights and
returns:

1. RMSE after NVFP4-style quantization.
2. RMSE after MXFP4-style quantization.

Implement both quantizers using NumPy only.

The FP4 codebook is:

$$
\{-1,-0.5,-0.25,-0.125,0,0.125,0.25,0.5,1\}.
$$

For NVFP4, split the tensor into consecutive blocks of 16 values. The block
scale is the maximum absolute value in the block divided by the largest
codebook magnitude.

For MXFP4, split the tensor into consecutive blocks of 32 values. The block
scale is restricted to a power of two. Choose the smallest power-of-two scale
that is greater than or equal to the NVFP4-style scale of that block.

Pad the final block with zeros when necessary. The returned RMSE values must be
Python floats.

## Example

```python
import numpy as np

w = np.array([0.0, 1.0, -0.5, 2.0], dtype=np.float32)
nv_rmse, mx_rmse = fp4_accuracy_comparison(w)
```

The exact values depend on the quantization procedure, but the NVFP4 RMSE should
not be larger than the MXFP4 RMSE for the provided evaluation weights.

## What the gate checks

The gate builds a deterministic weight tensor and computes the expected RMSE
values with an independent NumPy oracle implementing the same quantization
rules.

Your implementation passes when both returned RMSE values are within
$10^{-6}$ of the oracle values and the NVFP4 RMSE is strictly smaller than the
MXFP4 RMSE.
