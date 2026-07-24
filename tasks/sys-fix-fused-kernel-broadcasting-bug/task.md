## Context

Operator fusion combines multiple tensor operations into one kernel to reduce intermediate memory traffic. A fused implementation must preserve the same broadcasting semantics as the unfused operations.

Consider an input matrix $X \in \mathbb{R}^{n \times d}$, a row-scale vector $s \in \mathbb{R}^{d}$, and a bias vector $b \in \mathbb{R}^{d}$. The unfused computation is

$$
Y = X \odot s + b,
$$

where $\odot$ is element-wise multiplication and NumPy broadcasting expands $s$ and $b$ across the first dimension.

A fused kernel may accidentally treat the broadcast dimension as part of the element index and apply the vectors incorrectly. The output can have the correct shape but wrong values.

## Task

Implement `fused_affine(X, scale, bias)`:

```python
def fused_affine(X: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    ...
```

The inputs are:

- `X`: a 2-D NumPy array with shape $(n, d)$.
- `scale`: a 1-D NumPy array with shape $(d,)$.
- `bias`: a 1-D NumPy array with shape $(d,)$.

Return the fused result of

$$
Y_{ij} = X_{ij} \cdot scale_j + bias_j .
$$

The result must be a NumPy array with `float64` values. The implementation should preserve the broadcasting behavior of the unfused expression.

## Example

```python
import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
scale = np.array([2, 3, 4], dtype=np.float64)
bias = np.array([1, 0, -1], dtype=np.float64)

Y = fused_affine(X, scale, bias)
# [[ 3.  6. 11.]
#  [ 9. 15. 23.]]
```

## What the gate checks

The gate compares the implementation against a NumPy oracle that computes the unfused expression `X * scale + bias`.

The reported metric is the relative error

$$
\mathrm{rel\_err} =
\frac{\lVert Y_{\mathrm{candidate}} - Y_{\mathrm{oracle}} \rVert_2}
{\lVert Y_{\mathrm{oracle}} \rVert_2 + 10^{-12}} .
$$

The implementation passes when $\mathrm{rel\_err} \le 10^{-6}$ on several broadcast-shaped inputs.
