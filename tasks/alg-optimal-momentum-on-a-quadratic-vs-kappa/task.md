## Context

For a twice differentiable convex function $f:\mathbb{R}^n\to\mathbb{R}$ the gradient descent iteration with momentum is

$$x_{k+1}=x_k-\alpha\nabla f(x_k)+\beta (x_k-x_{k-1}),$$

where $\alpha>0$ is a stepsize and $0\le \beta<1$ is the momentum coefficient.  
When $f$ is a quadratic with Hessian $A$, i.e.

$$f(x)=\tfrac12 x^\top A\,x,$$

the eigenvalues of $A$ determine how fast the method converges.  Let

$$\lambda_{\min}\le \dots \le \lambda_{\max}$$

be the eigenvalues and define the condition number $\kappa=\lambda_{\max}/\lambda_{\min}>1$.  
For fixed stepsize $\alpha=2/(\lambda_{\min}+\lambda_{\max})$ the optimal momentum that minimises the asymptotic convergence factor is

$$\beta^{*}=\Bigl(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\Bigr)^2.$$

This value guarantees the fastest linear rate for all eigencomponents.

## Task

Implement a function `optimal_momentum_beta(A)` that receives a symmetric positive‑definite NumPy array `A` of shape `(n, n)`, computes its condition number $\kappa$, and returns the optimal momentum coefficient $\beta^{*}$ as a Python float.  The implementation must use only NumPy operations; no explicit loops are required.

```python
import numpy as np

def optimal_momentum_beta(A: np.ndarray) -> float:
    ...
```

The function should raise an informative `ValueError` if the input is not square or not positive‑definite.

## Example

```python
import numpy as np

A = np.array([[3, 1],
              [1, 2]], dtype=float)
beta = optimal_momentum_beta(A)
print(beta)   # ≈ 0.07179677
```

The returned value equals the analytical expression above for $\kappa=3/1=3$.

## What the gate checks

Two tests are performed:

* **Relative error** – The grader computes a reference $\beta^{*}$ from an oracle (NumPy eigenvalue routine) and compares your result with it.  Your implementation must satisfy  
  $$\frac{|\hat{\beta}-\beta^{*}|}{|\beta^{*}|}\le 10^{-6}.$$

* **Correctness of the contract** – The function must return a scalar `float` and accept only square NumPy arrays.

If either condition fails, the gate will not be passed.
