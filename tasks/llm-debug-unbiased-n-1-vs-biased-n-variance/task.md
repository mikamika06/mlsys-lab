## Context

Layer Normalization (LayerNorm) normalises the activations of a layer across its feature dimension.  
For an input vector $x \in \mathbb{R}^d$ we compute

$$\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,$$

and a variance estimate.  Two common choices exist:

* **Biased (population) variance**  
  $$\sigma_{\text{biased}}^{2}
    = \frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^2.$$

* **Unbiased (sample) variance**  
  $$\sigma_{\text{unbiased}}^{2}
    = \frac{1}{d-1}\sum_{j=1}^{d}(x_j-\mu)^2.$$

The normalised output is then

$$y_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}},$$

where $\varepsilon>0$ prevents division by zero.

In many deep‑learning libraries the default implementation uses the **biased** variance.  For this task you must implement a version that uses the **unbiased** variance (i.e. `ddof=1` in NumPy).

## Task

Implement the following function:

```python
def layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Apply Layer Normalization to a 2‑D array using unbiased variance.

    Parameters
    ----------
    x : np.ndarray of shape (n, d)
        Input activations.
    eps : float, optional
        Small constant added to the denominator for numerical stability.

    Returns
    -------
    y : np.ndarray of shape (n, d)
        Normalised activations.  The output must have dtype float64.
    """
```

The implementation must be fully vectorised with NumPy only – no Python loops.  
It should work for any 2‑D array of real numbers.

## Example

```python
import numpy as np

A = np.array([[1., 2., 3.],
              [4., 5., 6.]])
Y = layernorm(A, eps=0.)
print(Y)
# [[-1.22474487 -0.         1.22474487]
#  [-1.22474487 -0.         1.22474487]]
```

## What the gate checks

The grader generates several random test matrices and computes a reference output using NumPy’s unbiased variance (`np.var(..., ddof=1)`).  
It then compares your function’s result to this reference with the metric

$$\mathrm{max\_abs\_err} = \max_{i,j}\bigl|\,y^{\text{cand}}_{ij}-y^{\text{ref}}_{ij}\,\bigr|.$$

Your solution must achieve `max_abs_err <= 1e-6`.  Any deviation larger than this will fail the gate.
