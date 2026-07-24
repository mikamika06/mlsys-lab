## Context

In supervised learning the loss surface is often highly anisotropic, meaning that in some directions the curvature (second derivative) is much larger than in others. A canonical toy problem for studying this phenomenon is the quadratic objective

$$f(x)=\tfrac12\,x^\top A x,$$

where $A$ is a symmetric positive‑definite matrix with eigenvalues $\lambda_1 \le \dots \le \lambda_n$. The condition number $\kappa(A)=\lambda_{\max}/\lambda_{\min}$ measures how ill‑conditioned the problem is: large $\kappa$ implies that gradients point in directions of very different scales, which can slow down plain gradient descent.

Gradient descent (GD) updates

$$x_{t+1}=x_t-\eta\,\nabla f(x_t), \qquad \nabla f(x)=A x,$$

with a fixed step size $\eta$. For the quadratic case GD converges linearly if $0<\eta<2/\lambda_{\max}$. Adam is an adaptive‑momentum method that maintains running averages of gradients and squared gradients:

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)\nabla f(x_t),\\
v_t=\beta_2 v_{t-1}+(1-\beta_2)(\nabla f(x_t))^2,\\
x_{t+1}=x_t-\alpha\,\frac{m_t}{\sqrt{v_t}+\epsilon},
$$

with typical defaults $\beta_1=0.9,\;\beta_2=0.999,\;\alpha=10^{-3}$ and $\epsilon=10^{-8}$. Adam is known to be more robust on ill‑conditioned problems, but its convergence speed depends on the choice of $\alpha$.

## Task

Implement a function that runs both optimizers until the Euclidean norm of the gradient falls below a tolerance. The function must return the number of update steps taken by each optimizer.

```python
import numpy as np
from typing import Tuple

def sgd_vs_adam_steps(
    A: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 10000
) -> Tuple[int, int]:
    """
    Return the number of steps required for SGD and Adam to reach a gradient norm < tol.
    Parameters
    ----------
    A : (n, n) symmetric positive‑definite matrix defining f(x)=½xᵀAx
    x0 : initial point; if None defaults to an all‑ones vector
    tol : tolerance on ‖∇f‖₂
    max_iter : maximum number of iterations for each optimizer

    Returns
    -------
    sgd_steps, adam_steps : int
        Number of update steps performed by SGD and Adam respectively.
    """
```

The implementation must be deterministic: use `float64` everywhere, avoid randomness, and use the following hyper‑parameters:

* **SGD** – step size $\eta = 0.9 / \lambda_{\max}(A)$.
* **Adam** – learning rate $\alpha = 10^{-2}$, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$.

Both optimizers should stop as soon as the Euclidean norm of the gradient is strictly less than `tol`. If the tolerance is already satisfied at initialization, return `0` for that optimizer. Do **not** use any external libraries beyond NumPy.

## Example

```python
import numpy as np

# ill‑conditioned diagonal matrix with eigenvalues 1 and 1000
A = np.diag([1., 1000.])
x0 = np.ones(2)

sgd_steps, adam_steps = sgd_vs_adam_steps(A, x0)
print(sgd_steps, adam_steps)   # e.g. (1234, 56)
```

## What the gate checks

The grader runs a set of deterministic test cases with varying condition numbers and dimensions. For each case it computes the reference step counts using the same algorithm described above. The candidate’s output must match these integers exactly for **all** test cases; otherwise the `exact_match` metric fails.
