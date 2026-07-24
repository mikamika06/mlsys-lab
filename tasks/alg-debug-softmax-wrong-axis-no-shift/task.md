## Context

The standard **Softmax** function is defined as:

$$ \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} $$

A naive implementation that directly computes $e^{x_i}$ is numerically unstable. If the inputs $x_i$ are large (e.g., $x_i = 1000$), $e^{x_i}$ will overflow and evaluate to `inf`, resulting in `nan` after division.

To prevent overflow, we apply the **max-shift trick**. Since softmax is shift-invariant (shifting all elements by a constant $c$ doesn't change the output), we subtract the maximum value along the specified axis before exponentiating:

$$ \text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}} $$

## Task

The provided function `softmax(x, axis)` has two critical bugs:
1. It does not subtract the maximum value, causing numerical overflow.
2. It completely ignores the `axis` parameter, calculating the sum along the first dimension (`axis=0`) instead, which corrupts the output when the requested axis is different.

Your task is to debug and fix `softmax(x, axis)` so that it applies the max-shift correctly and respects the `axis` parameter.

## Example

```python
import numpy as np

x = np.array([[1000.0, 1001.0], [1000.0, 1000.0]])
out = softmax(x, axis=-1)

# Expected Output:
# [[0.26894142, 0.73105858],
#  [0.5       , 0.5       ]]
```

## What the gate checks

- `mean_kl`: Mean KL-divergence between the fixed implementation and the reference implementation must be $\le 10^{-9}$. We test on multi-dimensional inputs with large positive values to ensure the max-shift trick is applied.
