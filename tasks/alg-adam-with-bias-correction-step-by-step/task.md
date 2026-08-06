## Context

The Adam optimizer is a popular adaptive‑gradient method that combines ideas from momentum and RMSProp.  
For each parameter vector $\theta_t \in \mathbb{R}^d$ at iteration $t$, given a gradient $g_t = \nabla_{\theta}\!L(\theta_{t-1})$, Adam maintains two first‑order statistics:

$$
\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1-\beta_1)\, g_t ,\\[4pt]
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\, g_t^2 ,
\end{aligned}
$$

where $m_t$ is a biased estimate of the mean and $v_t$ of the uncentered variance.  
To correct for the bias introduced by initializing both moments at zero, Adam divides each statistic by its corresponding decay factor:

$$
\begin{aligned}
\hat m_t &= \frac{m_t}{1-\beta_1^t},\\[4pt]
\hat v_t &= \frac{v_t}{1-\beta_2^t}.
\end{aligned}
$$

The parameters are then updated with a step size $\eta$ (the learning rate) and a small constant $\varepsilon$ to avoid division by zero:

$$
\theta_t = \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}.
$$

This sequence of updates is repeated for each gradient in the training loop.

## Task

Implement a function that, given an initial parameter vector and a sequence of gradients, returns the full trajectory of parameters produced by Adam with bias correction. The signature must be:

```python
def adam_trajectory(
    params0: list[float],
    grads: list[list[float]],
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> list[list[float]]:
```

* `params0` – a list of floats of shape `(d,)`.
* `grads` – a two‑dimensional array of shape `(T, d)` where each row is the gradient at that step.
* The function should return an array of shape `(T+1, d)`.  The first row must be `params0`; subsequent rows are the parameters after each update.

The implementation uses plain Python; no external libraries are required.  All computations should be performed in `float64`.

## Example

```python

# Initial parameters and a toy gradient sequence
p0 = [0.0] * 3
g = [[1, 0, -1],
              [0, 2, 0]]

traj = adam_trajectory(p0, g)

print(traj)
```

Output (illustrative):

```
[[0.0, 0.0, 0.0], [-0.0009999999900000003, 0.0, 0.0009999999900000003], [-0.0016700582346581131, -0.0007441368183064559, 0.0016700582346581131]]
```

## What the gate checks

The grader computes a reference trajectory using the exact Adam algorithm described above with default hyper‑parameters and compares it to the candidate’s output.  
It reports the maximum absolute difference:

$$
\max_{i,j}\bigl|\, \text{candidate}_{ij} - \text{reference}_{ij}\,\bigr|.
$$

The solution must achieve a `max_abs_err` of at most $10^{-9}$.
