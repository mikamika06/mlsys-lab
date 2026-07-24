## Context

The objective function for a quadratic model is

$$f(x)=\tfrac12\,x^\top Q x - c^\top x,$$

where $Q \in \mathbb{R}^{d\times d}$ is symmetric positive‑definite and $c \in \mathbb{R}^d$.  
Its gradient is

$$\nabla f(x)=Qx-c.$$

For a smooth convex function with Lipschitz constant $L$ for the gradient, Nesterov’s accelerated gradient (NAG) achieves an optimal convergence rate of $\mathcal O(1/k^2)$ when the momentum parameter is chosen as

$$\beta=\frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}},$$

with $\mu$ the strong‑convexity constant (smallest eigenvalue of $Q$).  
The iterative scheme reads

$$
\begin{aligned}
y_k &= x_k + \beta(x_k-x_{k-1}),\\
x_{k+1} &= y_k - \alpha\,\nabla f(y_k),
\end{aligned}
$$

where $\alpha=1/L$ is a suitable step size.

## Task

Implement the function `nesterov_minimize` that performs $T$ iterations of NAG on the quadratic objective described above and returns the final iterate:

```python
def nesterov_minimize(Q: np.ndarray,
                      c: np.ndarray,
                      x0: np.ndarray,
                      lr: float,
                      beta: float,
                      T: int) -> np.ndarray:
    ...
```

The function must use only NumPy operations; no explicit Python loops over the dimension of $x$ are allowed. The returned array should have type `float64`.

## Example

```python
import numpy as np

# 2‑D quadratic with Q = [[3,1],[1,2]] and c = [1,0]
Q = np.array([[3., 1.], [1., 2.]])
c = np.array([1., 0.])

x0   = np.zeros(2)
lr   = 1./np.linalg.norm(Q, ord=2)          # 1/L
mu   = np.min(np.linalg.eigvalsh(Q))        # smallest eigenvalue
beta = (np.sqrt(lr**-1)-np.sqrt(mu))/(np.sqrt(lr**-1)+np.sqrt(mu))
T    = 20

x_T = nesterov_minimize(Q, c, x0, lr, beta, T)
print(x_T)   # close to the exact minimiser Q^{-1}c
```

## What the gate checks

The grader generates several random symmetric positive‑definite matrices $Q$, vectors $c$ and initial points $x_0$.  
For each case it computes a reference iterate using a trusted implementation of NAG, then evaluates the global relative error

$$\mathrm{rel\_err}=\frac{\|\,x_{\text{student}}-x_{\text{ref}}\|}{\|\,x_{\text{ref}}\|}.$$

The solution must achieve $\mathrm{rel\_err}\le 10^{-8}$ on all test cases.
