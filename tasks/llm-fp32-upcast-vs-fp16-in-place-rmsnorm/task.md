## Context

The RMSNorm of a vector $x \in \mathbb{R}^d$ is defined as  

$$\operatorname{rmsnorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}}.$$

When the input tensor is stored in half precision ($\mathrm{float16}$), computing the mean of $x^2$ directly in $\mathrm{float16}$ can lose significant accuracy because the intermediate sum may overflow or underflow. A common remedy is to **upcast** the reduction to a higher precision, typically $\mathrm{float32}$, before taking the square root and normalizing.

The task is to implement a vectorized RMSNorm that optionally performs this upcast for the reduction step while keeping the final output in $\mathrm{float16}$. The implementation must avoid explicit Python loops and rely solely on NumPy operations.

## Task

Implement the function

```python
def rmsnorm(x: np.ndarray, *, upcast: bool = True) -> np.ndarray:
    ...
```

* `x` is a 2‑D NumPy array of shape `(n, d)` with dtype `np.float16`.
* If `upcast=True`, cast the reduction to `float32` before computing the mean and square root.
* If `upcast=False`, perform all operations in `float16`.
* The function must return a new array of the same shape as `x` and dtype `np.float16`.

The implementation should be fully vectorized; no explicit Python loops are allowed.

## Example

```python
import numpy as np

A = np.array([[0, 0], [1, 0], [0, 2]], dtype=np.float16)
D = rmsnorm(A)          # upcast=True by default
print(D)
# [[ 0.   0.]
#  [ 1.   0.]
#  [ 0.   2.]]
```

## What the gate checks

The grader computes a reference RMSNorm using `float64` arithmetic for the reduction and compares it to the candidate's output with the metric  

$$\mathrm{max\_abs\_err} = \max_{i,j}\,|\,y^{\text{ref}}_{ij}-y^{\text{cand}}_{ij}\,|.$$

The gate requires `max_abs_err <= 0.001`. A correct implementation that uses the upcast reduction will satisfy this bound; a naive pure‑`float16` implementation typically fails.
