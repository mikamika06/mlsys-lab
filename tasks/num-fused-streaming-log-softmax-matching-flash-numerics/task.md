## Context

The softmax function converts logits $x \in \mathbb{R}^n$ into probabilities:

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} .
$$

Direct evaluation can overflow when logits contain large values. A numerically
stable log-softmax uses the maximum logit $m = \max_i x_i$:

$$
\log(\mathrm{softmax}(x_i)) =
x_i - m - \log\left(\sum_j e^{x_j-m}\right).
$$

Large-scale implementations often fuse normalization steps and avoid creating a
probability tensor. A streaming reduction can accumulate the shifted exponential
sum while keeping the computation stable:

$$
s = \sum_i e^{x_i-m},
$$

then reuse the scalar value $\log(s)$ for every output element.

## Task

Implement `streaming_log_softmax(x)`:

```python
def streaming_log_softmax(x: np.ndarray) -> np.ndarray:
    ...
```

The function takes a one-dimensional NumPy array of logits and returns a
float64 NumPy array containing $\log(\mathrm{softmax}(x))$.

Use max-subtraction before exponentiation. The implementation must not compute an
unstable expression such as `np.exp(x) / np.sum(np.exp(x))` because this can
overflow for valid inputs.

## Example

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = streaming_log_softmax(x)

# approximately:
# [-2.40760596, -1.40760596, -0.40760596]
```

## What the gate checks

The gate computes a float64 NumPy oracle using the stable log-softmax formula and
compares the returned values against it.

The metric `max_abs_err` is:

$$
\max_i |y_i - y_i^{\mathrm{oracle}}|.
$$

The value must be less than $10^{-6}$. Test cases include logits with very large
magnitudes where an implementation that materializes unshifted exponentials
produces overflow or invalid values.
