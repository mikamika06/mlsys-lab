## Context

A quadratic objective function in $\mathbb{R}^n$ has the form  

$$f(x) = \tfrac12\,x^\top A x - b^\top x,$$

where $A\in\mathbb{R}^{n\times n}$ is symmetric positive‑definite and $b\in\mathbb{R}^n$.  
Its gradient is linear:

$$\nabla f(x) = A\,x - b.$$

Gradient descent with a momentum term (the heavy‑ball method) updates an auxiliary velocity vector $v$ and the iterate $x$ as

$$
\begin{aligned}
v_{t+1} &= \beta\,v_t - \eta\,\nabla f(x_t),\\
x_{t+1} &= x_t + v_{t+1},
\end{aligned}
$$

with learning rate $\eta>0$ and momentum coefficient $0\le\beta<1$.  
When $\beta=0$ the method reduces to vanilla gradient descent.

## Task

Implement a function that performs $T$ iterations of SGD with momentum on the quadratic objective described above. The function must have the following signature:

```python
def sgd_momentum_quadratic(
    A: np.ndarray,
    b: np.ndarray,
    init_x: np.ndarray,
    lr: float,
    momentum: float,
    T: int
) -> np.ndarray:
    ...
```

* `A` – a symmetric positive‑definite matrix of shape $(n,n)$.
* `b` – a vector of shape $(n,)`.
* `init_x` – the starting point, shape $(n,)$.
* `lr` – learning rate $\eta$.
* `momentum` – momentum coefficient $\beta$.
* `T` – number of iterations.

The function should return the final iterate $x_T$.  
All computations must use NumPy; no explicit Python loops over the dimension $n$, but a loop over $T$ is allowed. The returned array must be of type `float64`.

## Example

```python
import numpy as np

A = np.array([[2., 0.], [0., 3.]])
b = np.array([1., 2.])
init_x = np.zeros(2)
lr = 0.1
momentum = 0.9
T = 50

x_T = sgd_momentum_quadratic(A, b, init_x, lr, momentum, T)
print(x_T)   # approximately [0.5, 0.66666667]
```

## What the gate checks

The grader computes a reference iterate using the exact heavy‑ball update described above on several randomly generated test cases.  
Your implementation must produce an output whose relative L2 error satisfies  

$$\frac{\|x_{\text{cand}}-x_{\text{ref}}\|_2}{\|x_{\text{ref}}\|_2}\le 10^{-8}.$$

The gate metric is named `rel_err`. A failure indicates a logic error in the update equations or incorrect handling of the momentum term.
