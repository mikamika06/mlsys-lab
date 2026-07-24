## Context
Computing the variance of a dataset using the naive formula $\text{Var}(X) = E[X^2] - (E[X])^2$ is prone to catastrophic cancellation when the variance is small compared to the squared mean. 
Welford's online algorithm provides a numerically stable way to compute the sample or population variance in a single pass.

The recurrence relations for Welford's algorithm are:
$$ \mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n} $$
$$ M_{2,n} = M_{2,n-1} + (x_n - \mu_{n-1})(x_n - \mu_n) $$

Where $M_{2,n}$ tracks the sum of squared differences from the current mean. The population variance is given by $M_{2,n} / n$.

## Task
Implement the function `welford_variance(data)` that computes the population variance of a given list of floats using Welford's algorithm.

Your implementation must be numerically stable even when the data has a very large mean and a very small variance.

## Example
```python
data = [1e9 + 1, 1e9 + 2, 1e9 + 3]
welford_variance(data)
# 0.6666666666666666
```

## What the gate checks
- `rel_err`: Relative error compared to the true variance (computed using higher precision or numerically stable oracle like `numpy.var`). It must be $\le 10^{-8}$.
