## Context

Gradient descent is the backbone of many machine learning algorithms.  
Given a differentiable loss function $L(\theta)$, its gradient $\nabla_\theta L$ points in the direction of steepest ascent; moving opposite to this vector reduces the loss.

The vanilla update rule for stochastic gradient descent (SGD) is

$$
\theta_{t+1} \;\gets\; \theta_t - \eta\, g_t,
$$

where $g_t = \nabla_\theta L(\theta_t)$ and $\eta>0$ is the learning rate.  
In practice, SGD can suffer from noisy gradients or ill‑conditioned curvature, which motivates adaptive methods such as RMSProp.

RMSProp maintains a running average of squared gradients:

$$
v_{t+1} \;=\; \rho\, v_t + (1-\rho)\, g_t^2,
$$

with decay rate $\rho\in(0,1)$.  
The parameter update then scales the gradient by the root‑mean‑square of past gradients:

$$
\theta_{t+1}
  \;\gets\;
  \theta_t - \frac{\eta}{\sqrt{v_{t+1}} + \varepsilon}\; g_t,
$$

where $\varepsilon>0$ prevents division by zero.  
This scheme automatically reduces the step size for frequently large gradients, stabilising training.

## Task

Implement a function that, given a sequence of gradient vectors, returns the trajectory of parameters produced by RMSProp.

```python
import numpy as np

def rmsprop_trajectory(
    grads: np.ndarray,
    lr: float = 0.01,
    eps: float = 1e-8,
    decay_rate: float = 0.9
) -> np.ndarray:
    """
    Compute the RMSProp trajectory for a sequence of gradients.

    Parameters
    ----------
    grads : (T, d) array_like
        Sequence of gradient vectors; each row is g_t.
    lr : float, optional
        Learning rate η.
    eps : float, optional
        Small constant added to denominator.
    decay_rate : float, optional
        Decay rate ρ for the squared‑gradient accumulator.

    Returns
    -------
    trajectory : (T+1, d) ndarray of dtype float64
        Parameter vectors θ_0 … θ_T.  The initial vector is all zeros.
    """
    ...
```

The function must use only NumPy operations; no Python loops over dimensions are allowed in the user code.

## Example

```python
import numpy as np

grads = np.array([[1., 2.],
                  [0.5, -1.],
                  [-0.3, 0.8]])

traj = rmsprop_trajectory(grads, lr=0.1)
print(traj)
```

Output (rounded to 4 decimals):

```
[[ 0.0000  0.0000]
 [-0.1000 -0.2000]
 [-0.1455 -0.2800]
 [-0.1519 -0.2728]]
```

## What the gate checks

The grader computes a reference trajectory using NumPy and compares it to your output with the scorer `max_abs_err`.  
Your implementation must satisfy

$$
\max_{t,i} |\, \theta^{\text{your}}_{t,i} - \theta^{\text{ref}}_{t,i}\,| \;\le\; 10^{-9}.
$$

Any deviation larger than this threshold will cause the gate to fail. The grader also verifies that your function returns a NumPy array of dtype `float64` and shape `(T+1, d)`.
