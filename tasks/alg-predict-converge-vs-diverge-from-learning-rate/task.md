## Context

Gradient descent updates parameters by moving opposite to the gradient:

$$
x_{t+1} = x_t - \alpha \nabla f(x_t),
$$

where $\alpha$ is the learning rate. For a convex quadratic objective with gradient Lipschitz constant $L$, the learning rate controls whether repeated updates are stable.

The basic convergence condition is:

$$
0 < \alpha < \frac{2}{L}.
$$

Learning rates at or above this boundary are expected to diverge because the update can overshoot instead of reducing the error.

## Task

Implement `classify_learning_rates(lrs, L)`:

```python
def classify_learning_rates(lrs, L):
    ...
```

The function receives a list of learning rates and a positive gradient Lipschitz constant $L$. Return a list of integers with the same length.

For each learning rate $\alpha$:
- return `0` if $\alpha < \frac{2}{L}$ (predicted to converge),
- return `1` if $\alpha \ge \frac{2}{L}$ (predicted to diverge).

The output must contain only Python integers.

## Example

```python
lrs = [0.01, 0.1, 0.2, 0.25]
L = 10

classify_learning_rates(lrs, L)
# [0, 0, 0, 1]
```

## What the gate checks

The gate computes the reference classification from the mathematical gradient descent stability boundary $\frac{2}{L}$ and compares the returned list exactly. Boundary values are included, so using a strict comparison in the wrong direction will fail.
