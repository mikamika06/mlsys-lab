## Context

The condition number of a square matrix $A \in \mathbb{R}^{n \times n}$ measures how
sensitive $Ax = b$ is to perturbations in $b$.  A large condition number means
small input changes can produce large output changes — the system is
*ill-conditioned*.

When $A$ is invertible, the 2-norm condition number is defined as

$$\kappa(A) = \lVert A \rVert_2 \cdot \lVert A^{-1} \rVert_2
           = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular
values returned by the singular value decomposition.  The identity follows
because $\lVert A \rVert_2 = \sigma_{\max}$ and
$\lVert A^{-1} \rVert_2 = 1/\sigma_{\min}$.

NumPy provides `np.linalg.cond(A)` which computes this quantity.  Your task is
to reproduce the result by performing the SVD yourself.

## Task

Implement `condition_number_via_svd(A)`:

```python
import numpy as np

def condition_number_via_svd(A: np.ndarray) -> float:
    """Return the 2-norm condition number of square matrix A.
    
    Compute the SVD via numpy and return sigma_max / sigma_min as a float.
    """
```

The function takes a 2-D square NumPy array and returns a single `float`.  Use
`np.linalg.svd` (or any NumPy SVD routine) — do not call `np.linalg.cond`
directly.

## Example

```python
import numpy as np
A = np.array([[2.0, 0.0],
              [0.0, 0.5]])
kappa = condition_number_via_svd(A)
# kappa == 4.0  because sigma_max=2, sigma_min=0.5
```

A near-singular matrix gives a very large condition number:

```python
B = np.array([[1.0, 0.0],
              [0.0, 1e-14]])
kappa = condition_number_via_svd(B)
# kappa == 1e14
```

## What the gate checks

A single gate: the relative error $\mathrm{rel\_err}$ against the NumPy
oracle `np.linalg.cond(A)` must satisfy

$$\mathrm{rel\_err} < 10^{-6}$$

where

$$\mathrm{rel\_err} = \frac{\bigl|\kappa_{\text{yours}} - \kappa_{\text{ref}}\bigr|}{|\kappa_{\text{ref}}| + \epsilon}\,.$$

The test cases include well-conditioned, ill-conditioned, near-singular, and
random matrices.  A naive implementation that calls `np.linalg.cond` directly
would also pass numerically, but the reference solution and starter are
structured so the learner must write the SVD-then-divide logic themselves.
