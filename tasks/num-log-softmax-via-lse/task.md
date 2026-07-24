## Context

The softmax function maps a vector $x \in \mathbb{R}^n$ to a probability distribution:

$$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)}$$

In many applications — cross-entropy loss, attention mechanisms, the categorical
reparameterization trick — we need the **log** of softmax, not softmax itself.
The naïve approach computes

$$\text{log\_softmax}(x_i) = \log\!\left(\frac{\exp(x_i)}{\sum_j \exp(x_j)}\right) = x_i - \log\!\sum_j \exp(x_j)$$

but this is numerically dangerous. When any component $x_j$ exceeds roughly $709$,
$\exp(x_j)$ overflows `float64` to $+\infty$, and when all components are large
and negative, every $\exp(x_j)$ underflows to $0$, producing $\log(0) = -\infty$.

The **log-sum-exp (LSE)** trick fixes this. Let $m = \max_j x_j$. Then:

$$\log\sum_j \exp(x_j) \;=\; m \;+\; \log\!\sum_j \exp(x_j - m)$$

Because every $x_j - m \le 0$, the exponentials are at most $\exp(0) = 1$ and
never overflow. At least one term equals $1$, so the sum is never zero and the
logarithm is always finite. The stable log-softmax is therefore:

$$\text{log\_softmax}(x_i) = x_i - m - \log\!\sum_j \exp(x_j - m)$$

## Task

Implement `log_softmax`:

```python
def log_softmax(x: np.ndarray) -> np.ndarray:
    ...
```

**Input:** a 1-D NumPy array of `float64` values, length $n \ge 1$.

**Output:** a 1-D `float64` array of the same length containing

$$\text{log\_softmax}(x_i) = x_i - m - \log\!\sum_{j=1}^{n}\exp(x_j - m), \qquad m = \max_j x_j.$$

Use only NumPy vectorized operations — no Python `for` loops. The result must
satisfy $\max_i |y_i - y_i^{\text{ref}}| < 10^{-7}$ where $y^{\text{ref}}$ is
the reference stable log-softmax.

## Example

```python
import numpy as np
x = np.array([0.0, 1.0, 2.0, 3.0])
y = log_softmax(x)
# y ≈ [-3.4402, -2.4402, -1.4402, -0.4402]
# sum(exp(y)) ≈ 1.0  (valid probability distribution in log-space)
```

A naïve `x - np.log(np.sum(np.exp(x)))` gives the same answer here, but it
**overflows** on the test vector `[1000.0, 1001.0, 1002.0]`, while the stable
formula handles it without issue.

## What the gate checks

The gate reports `max_abs_err`, the worst-case absolute error over all test
vectors. It is computed as:

$$\text{max\_abs\_err} = \max_{\text{cases}} \max_i \left| y_i - y_i^{\text{ref}} \right|$$

where each $y^{\text{ref}}$ is the NumPy stable formula $x - m - \log\sum_j \exp(x_j - m)$.
The gate passes when `max_abs_err` $< 10^{-7}$.

Test vectors include:
- moderate values (`[0, 1, 2, 3]`) — sanity check
- large positive values (`[1000, 1001, 1002]`) — naïve `exp` overflows
- large negative values (`[-1000, -1001, -1002]`) — naïve `exp` underflows
- mixed extreme range (`[-500, 0, 500]`)
- uniform values (`[1, 1, 1]`)
- single element (`[0]`)
