## Context

Low-bit weight quantization replaces floating point values with a small discrete grid. A common production technique for neural network weights is NormalFloat4 (NF4), which chooses a non-uniform 4-bit codebook based on the normal distribution. A tensor is quantized by mapping each value to its nearest codebook entry and then reconstructing it.

For a tensor $W$ and reconstructed tensor $\hat{W}$, the reconstruction mean squared error is

$$
\mathrm{MSE}(W, \hat{W}) = \frac{1}{N}\sum_{i=1}^{N}(W_i-\hat{W}_i)^2 .
$$

This task compares three 4-bit grids:

- NF4: a normal-distribution-derived non-uniform codebook.
- FP4: a fixed floating-point-style 4-bit codebook.
- int4-affine: a uniformly spaced affine integer grid with learned scale and zero point.

The quantization process is a nearest-neighbor projection:

$$
q(x)=\arg\min_{c \in C}|x-c|,
$$

where $C$ is the grid of representable values.

## Task

Implement `quantization_mse_triplet(W)`:

```python
def quantization_mse_triplet(W: np.ndarray) -> tuple[float, float, float]:
    ...
```

The function receives a one-dimensional NumPy array of normally distributed weights and returns:

```python
(nf4_mse, fp4_mse, int4_affine_mse)
```

The implementation must:

1. Quantize and reconstruct the input using the NF4 codebook.
2. Quantize and reconstruct the input using the FP4 codebook.
3. Quantize and reconstruct the input using an affine int4 grid with 16 integer levels.
4. Return the three reconstruction MSE values as Python floats.

Use NumPy operations for the numeric computation.

## Example

```python
import numpy as np

W = np.array([-1.0, -0.2, 0.0, 0.7], dtype=np.float64)

nf4, fp4, int4 = quantization_mse_triplet(W)

# Each value is a reconstruction error.
print(nf4, fp4, int4)
```

The exact values depend on the input distribution.

## What the gate checks

The gate generates a deterministic normally distributed weight tensor and computes the oracle result using the same quantization algorithms with NumPy.

The returned vector of three MSE values is compared using relative error:

$$
\mathrm{rel\_err} =
\frac{\lVert y-\hat{y}\rVert_2}{\lVert y\rVert_2+\epsilon}.
$$

The oracle also verifies the expected ordering on the generated normal weights:

$$
\mathrm{MSE}_{NF4} < \mathrm{MSE}_{FP4} < \mathrm{MSE}_{int4\text{-}affine}.
$$

A submission that returns incorrect grids or uses one quantizer for all three outputs will fail.
