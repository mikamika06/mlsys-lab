## Context

In transformer architectures residual connections are added after each sub‑module. When training very deep stacks the norm of the accumulated residual can grow or shrink exponentially, which hurts convergence. DeepNorm proposes to scale the residual at every block by a factor that depends on the depth and a hyper‑parameter $\alpha$. For a stack of $L$ blocks with residuals $r_0,\dots,r_{L-1}$ the scaled norm at layer $i$ is

$$\hat{n}_i = \lVert r_i\rVert_2\,\alpha^i,$$

where $\alpha>0$ controls how aggressively the norms are damped or amplified.  The sequence $\{\hat n_i\}$ can be used as a diagnostic of stability: if it grows too fast the network is likely to explode, if it decays too fast gradients may vanish.

## Task

Implement `deepnorm_scaled_residuals(residuals, alpha)`:

```python
def deepnorm_scaled_residuals(residuals: list[np.ndarray], alpha: float) -> np.ndarray:
    ...
```

`residuals` is a list of 1‑D NumPy arrays, one per transformer block.  
`alpha` is a positive scalar. The function must return a 1‑D array of length `len(residuals)` containing the scaled L2 norms $\hat n_i$ defined above. All computations should be performed in float64.

## Example

```python
import numpy as np
res = [np.array([1,0]), np.array([0,2])]
alpha = 2.0
print(deepnorm_scaled_residuals(res, alpha))
# [1.0, 4.0]
```

Here $\lVert[1,0]\rVert_2=1$ and $\lVert[0,2]\rVert_2=2$, so the scaled norms are $1\cdot2^0=1$ and $2\cdot2^1=4$.

## What the gate checks

The grader generates random residuals of varying length and dimensionality.  
For each case it computes a reference answer with the same algorithm in double precision.  
Your implementation must produce an array whose global relative L2 error satisfies

$$\mathrm{rel\_err} = \frac{\lVert \hat n - \hat n_{\text{ref}}\rVert}{\lVert \hat n_{\text{ref}}\rVert}\le 10^{-2}.$$

If the error exceeds this threshold on any test case the solution fails.
