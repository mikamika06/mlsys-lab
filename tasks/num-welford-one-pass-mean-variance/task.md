## Context

Welford's algorithm computes the mean and (population) variance of a stream
in a **single pass**, touching each element exactly once, without ever
forming a two-pass intermediate or a large squared sum. Starting from
$n_0 = 0,\ \bar x_0 = 0,\ M_{2,0} = 0$, each new sample $x_i$ ($i = 1,
2,\dots$) updates:

$$
n_i = n_{i-1} + 1, \qquad
\delta_i = x_i - \bar x_{i-1}, \qquad
\bar x_i = \bar x_{i-1} + \frac{\delta_i}{n_i}, \qquad
$$
$$
\delta_i' = x_i - \bar x_i, \qquad
M_{2,i} = M_{2,i-1} + \delta_i\,\delta_i' .
$$

After all $N$ samples, $\bar x_N$ is the mean and $M_{2,N}/N$ is the
population variance. Because the running mean is updated *before* it is used
to compute $M_2$'s increment, this recurrence never needs $\sum x_i^2$ —
which is exactly the quantity that blows up and cancels catastrophically when
the data sits far from zero (see the naive one-pass formula
$\operatorname{Var} = \mathbb{E}[X^2] - \mathbb{E}[X]^2$).

## Task

Implement:

```python
def welford_mean_var(x: np.ndarray) -> tuple[float, float]:
    ...
```

* `x` — a 1-D `float64` NumPy array.
* Return `(mean, var)`: the mean and population variance of `x`, computed by
  iterating over `x` **element by element** and applying the update
  recurrence above — not by calling `np.mean`/`np.var`/`np.sum(x**2)` or any
  other whole-array reduction, and not by making two separate passes over
  `x`. Each element must be visited exactly once.

## Example

```python
import numpy as np

x = np.array([4.0, 8.0, 15.0, 16.0, 23.0, 42.0])
mean, var = welford_mean_var(x)
print(mean, var)
# 18.0  147.33333333333334
```

## What the gate checks

* `rel_err` — the grader builds a fixture with a large common offset (mean
  around $10^{6}$, so the naive $\mathbb{E}[X^2]-\mathbb{E}[X]^2$ formula
  would already be losing digits) and compares your `(mean, var)` to an
  independent two-pass `float64` reference
  (`np.mean(x)`, `np.mean((x - mean)**2)`) with `rel_err`. Gate: `<= 1e-9`.
* `line_events` — the grader runs `arena.probe.count_line_events` on your
  function to count executed Python-level bytecode lines. A genuine
  element-by-element loop over hundreds of samples emits thousands of line
  events; a vectorized `np.mean`/`np.var` shortcut emits only a handful
  regardless of input size. Gate: `line_events >= 1000` — this is what
  actually enforces the "single pass, no whole-array reduction" requirement
  above.
