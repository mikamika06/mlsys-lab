## Context

The squared Euclidean distance between two vectors $a,b\in\mathbb{R}^d$ is defined as

$$\lVert a-b\rVert^2 = \sum_{i=1}^{d}(a_i-b_i)^2.$$

A useful algebraic identity expands this expression:

$$\lVert a-b\rVert^2
   = \lVert a\rVert^2 + \lVert b\rVert^2 - 2\,a^\top b,$$

where $\lVert a\rVert^2=\sum_i a_i^2$ and $a^\top b=\sum_i a_i b_i$.  
The identity is exact for all real vectors; it follows directly from expanding the square.

## Task

Implement the function `sq_dist_expansion(a, b)` that takes two 1‑dimensional NumPy arrays of equal length and returns their squared Euclidean distance computed **only** with the expansion formula above. The result must be a Python float (or a NumPy scalar) of type `float64`.

```python
def sq_dist_expansion(a: np.ndarray, b: np.ndarray) -> float:
    ...
```

Do not use any explicit loops or NumPy operations that directly compute $(a-b)^2$; rely solely on dot products and norms.

## Example

```python
import numpy as np
from your_module import sq_dist_expansion

a = np.array([1.0, 3.0, -2.0])
b = np.array([4.0, 0.0, 5.0])

dist_sq = sq_dist_expansion(a, b)
print(dist_sq)          # 49.0
```

The naive computation `np.sum((a-b)**2)` also yields `49.0`.

## What the gate checks

The grader evaluates your implementation on several random vector pairs and compares it to a NumPy reference that uses the straightforward definition $\sum (a_i-b_i)^2$.  
It reports the maximum absolute error via the scorer `max_abs_err`.  Your solution must satisfy

$$\mathrm{max\_abs\_err} \le 10^{-9}.$$

Any larger deviation causes the gate to fail.
