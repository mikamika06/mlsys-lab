## Context

Given $n$ independent observations with per-observation probabilities
$p_1, p_2, \dots, p_n \in (0, 1]$, the likelihood of the whole dataset under a
model is the product

$$
L = \prod_{i=1}^{n} p_i .
$$

The log-likelihood is $\log L$. A direct implementation multiplies the
probabilities first and takes the logarithm afterward:

$$
\log L = \log\left(\prod_{i=1}^{n} p_i\right) .
$$

This is numerically unsafe. Each $p_i < 1$ shrinks the running product, and in
IEEE-754 double precision the smallest positive normal magnitude is about
$10^{-308}$. For a few thousand observations the product underflows to
exactly $0.0$ long before the loop finishes, and $\log(0) = -\infty$ — the
likelihood computation silently collapses regardless of the true value.

The fix is to move the logarithm inside the product, turning multiplication
into addition:

$$
\log L = \log\left(\prod_{i=1}^{n} p_i\right) = \sum_{i=1}^{n} \log p_i .
$$

Summing $n$ numbers of moderate magnitude never underflows the way a product
of $n$ shrinking factors does, so this form stays accurate for arbitrarily
long sequences of probabilities.

## Task

Fix `log_likelihood(probs)`:

```python
def log_likelihood(probs: np.ndarray) -> float:
    ...
```

The input is a 1-D NumPy array of probabilities, each in $(0, 1]$. Return the
log-likelihood as a Python `float`, computed as the sum of the elementwise
logarithms rather than the log of the product.

The current implementation is:

```python
def log_likelihood(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    return float(np.log(np.prod(probs)))
```

This matches the mathematical definition but underflows for long arrays of
small probabilities, returning `-inf` instead of the true (finite) value.

## Example

```python
import numpy as np
probs = np.full(3000, 0.5, dtype=np.float64)
log_likelihood(probs)
# correct: -3000 * log(2) ≈ -2079.44
# broken:  np.prod(probs) underflows to 0.0 -> log(0.0) = -inf
```

## What the gate checks

The grader evaluates `log_likelihood` on several arrays of probabilities,
including ones long enough that their raw product underflows to `0.0` in
float64. It compares the returned value to a reference computed as
$\sum_i \log p_i$ in float64 using the relative error

$$
\mathrm{rel\_err} = \frac{|\hat{y} - y|}{|y| + 10^{-12}} .
$$

The gate requires $\mathrm{rel\_err} \le 10^{-10}$ on every test case. The
`np.log(np.prod(...))` form returns `-inf` on the underflowing case, which is
non-finite and fails the gate immediately.
