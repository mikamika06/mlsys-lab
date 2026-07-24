## Context

Vectors $s, t \in \mathbb{R}^n$ can be sorted elementwise. The sorting operation $y = \text{sort}(s)$ is piecewise linear and its Jacobian is the permutation matrix $P$ that moves the elements into sorted order. For a differentiable loss $L(y)$ defined on the sorted vector, the gradient with respect to $s$ is obtained by applying the inverse permutation:

$$\frac{\partial L}{\partial s} = P^\top \frac{\partial L}{\partial y}.$$

A concrete loss that relies on sorting is the **Unsorted Label Distance (ULD)**:

$$\mathcal{L}_{\text{ULD}}(s, t) = \|\text{sort}(s) - \text{sort}(t)\|_2^2,$$

where $s$ are student logits and $t$ are teacher logits.

## Task

Implement the function

```python
def uld_gradient(student_logits: np.ndarray, teacher_logits: np.ndarray) -> np.ndarray:
```

that returns the gradient $\frac{\partial \mathcal{L}_{\text{ULD}}}{\partial s}$.

- Both inputs are 1-D `numpy.ndarray` of equal length.
- Return a 1-D `numpy.ndarray` of the same shape (float64).

## Example

```python
import numpy as np
s = np.array([1.0, 3.0, 2.0])
t = np.array([0.0, 4.0, 1.0])
grad = uld_gradient(s, t)
# Expected:
#   sort(s) = [1, 2, 3],  sort(t) = [0, 1, 4]
#   diff = [1, 1, -1]
#   rank = np.argsort(s).argsort()  = [0, 2, 1]
#   grad = 2 * diff[rank] = [2, -2, 2]
print(grad)   # [ 2. -2.  2.]
```

## What the gate checks

The checker computes the true gradient by central finite differences on $\mathcal{L}_{\text{ULD}}$ with $\epsilon = 10^{-5}$. Your result must have a maximum absolute error smaller than $1 \times 10^{-5}$ compared to this oracle. The test cases are generated from random continuous values (no ties), so any consistent tie-breaking subgradient is acceptable.
