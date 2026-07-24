## Context

RoPE (Rotary Position Embedding) encodes absolute positions by rotating embedding vectors with a base angle $\theta$. For a position $p$ and dimension index $k$, the rotation angle is
$$\phi_{p,k} = \theta^{\,p/k_{\text{dim}}},$$
where $k_{\text{dim}}$ is a scaling constant. When extending a model to longer contexts than it was trained on, the effective base $\theta$ must be adjusted so that the relative geometry of positions remains consistent with the training NTK (Neural Tangent Kernel). A simple NTK‑aware adjustment rescales the base by a factor $f>0$:
$$\theta_{\text{new}} = \theta_{\text{old}}^{\,f}.$$
This preserves the logarithmic spacing of angles while scaling the overall magnitude.

## Task

Implement `scale_rope_base(theta, factor)`:

```python
def scale_rope_base(theta: np.ndarray | float,
                    factor: np.ndarray | float) -> np.ndarray:
    ...
```

The function should accept scalars or NumPy arrays and return a NumPy array of type `float64`. It must compute the element‑wise exponentiation $\theta^{\,\text{factor}}$.

## Example

```python
import numpy as np
from solution_ref import scale_rope_base

theta = np.array([1.0, 2.0, 4.0])
f = 0.5
print(scale_rope_base(theta, f))
# [1.         1.41421356 2.        ]
```

## What the gate checks

The grader generates a set of test pairs $(\theta,\text{factor})$ and compares your implementation against the reference computed by NumPy using $\theta^{\,\text{factor}}$. It uses the scorer
```python
from arena.scorers import max_abs_err
```
and passes if the maximum absolute error is at most $10^{-6}$.
