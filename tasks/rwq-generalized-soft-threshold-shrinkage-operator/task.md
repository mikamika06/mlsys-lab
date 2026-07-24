## Context

The proximal operator of the $L_p$ quasi‑norm for $0<p\le1$ is defined by

$$
\\operatorname{prox}_{\\lambda \\|\\cdot\\|_p^p}(v)
   = \\arg\\min_{x}\\;\\frac12\\lVert x-v\\rVert_2^2+\\lambda\\,\\lvert x\\rvert^p .
$$

For $p=1$ this reduces to the familiar soft‑thresholding rule
$\\operatorname{sign}(v)\\,\\max(|v|-\\lambda,0)$.  
When $p<1$ the shrinkage depends on the magnitude of each component:
for a scalar $t\\in\\mathbb R$

$$
\\operatorname{prox}_{\\lambda |\\,\\cdot\\,|^p}(t)
   = \\operatorname{sign}(t)\\,\\max\\bigl(|t|-\\,\\lambda\\,|t|^{\,p-1},\\,0\\bigr).
$$

This is the *generalized soft‑threshold* operator that we ask you to implement.

## Task

Implement `generalized_soft_threshold(x, beta, p)`:

```python
def generalized_soft_threshold(x: np.ndarray, beta: float, p: float) -> np.ndarray:
    ...
```

The function receives a 1‑D NumPy array `x`, a positive scalar `beta` (the regularisation strength), and a quasi‑norm exponent `p` with $0<p\\le1$.  
It must return an array of the same shape containing the element‑wise shrinkage described above. The implementation must use only NumPy operations; no explicit Python loops.

## Example

```python
import numpy as np
x = np.array([3.0, -2.5, 0.4])
beta = 1.0
p = 0.7
y = generalized_soft_threshold(x, beta, p)
# y ≈ [ 2.0, -1.5, 0. ]
```

## What the gate checks

The grader evaluates your implementation against a NumPy reference on several random test vectors.  
It reports the maximum absolute error

$$\\max_i |\\,y_{\\text{your}}[i]-y_{\\text{ref}}[i]\\,|.$$

Your solution must achieve an error not larger than $10^{-8}$.
