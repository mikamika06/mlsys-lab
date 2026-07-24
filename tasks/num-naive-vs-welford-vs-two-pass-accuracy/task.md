## Context

The population variance of $x_1, \dots, x_n$ is

$$
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2, \qquad \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i .
$$

A common shortcut expands the square algebraically to avoid a second pass
over the data:

$$
\sigma^2 \;\overset{\text{naive}}{=}\; \frac{1}{n}\sum_i x_i^2 - \bar{x}^2 = \mathbb{E}[X^2] - \mathbb{E}[X]^2 .
$$

This "naive" one-pass formula is a textbook identity, but it is numerically
dangerous: when the data has a large mean relative to its spread (for
example, values clustered around $10^8$ with unit variance), both
$\mathbb{E}[X^2]$ and $\mathbb{E}[X]^2$ are huge numbers of nearly equal
magnitude ($\approx 10^{16}$). Subtracting two nearly equal floats destroys
almost every significant digit — the result can be wildly wrong, or even
negative, which is impossible for a true variance.

Welford's online algorithm avoids this cancellation entirely by updating a
running mean and a running sum of squared deviations $M_2$ incrementally,
one sample at a time:

$$
\delta = x_i - \bar{x}_{i-1}, \qquad
\bar{x}_i = \bar{x}_{i-1} + \frac{\delta}{i}, \qquad
\delta' = x_i - \bar{x}_i, \qquad
M_{2,i} = M_{2,i-1} + \delta \cdot \delta' ,
$$

starting from $\bar{x}_0 = 0, M_{2,0} = 0$. After processing all $n$ samples,
$\sigma^2 = M_{2,n} / n$. Because every update works with differences from
the current running mean rather than raw squared magnitudes, it never forms
the huge intermediate values that make the naive formula unstable. A
two-pass approach (compute the mean first, then sum $(x_i - \bar{x})^2$ using
the already-known mean) is also stable for this kind of data, since it
never needs the naive expansion either — but Welford does it online, in a
single pass, without storing the whole array in a second traversal.

## Task

Implement `welford_variance(data)`:

```python
def welford_variance(data: np.ndarray) -> float:
    ...
```

`data` is a 1-D `float64` NumPy array. Return the population variance
(divide by $n$, not $n-1$) computed with Welford's online algorithm — track a
running mean and running $M_2$ while iterating over the array once, and
return $M_2 / n$ at the end. Do not use the naive
$\mathbb{E}[X^2] - \mathbb{E}[X]^2$ expansion.

## Example

```python
import numpy as np
x = np.array([1e8 + 0.5, 1e8 - 0.5, 1e8 + 1.0, 1e8 - 1.0])
welford_variance(x)
# ~= 0.625   (the variance of the small offsets, unaffected by the 1e8 shift)
```

## What the gate checks

The grader loads a fixture of 20000 samples drawn from a unit-variance normal
distribution shifted by $10^8$ — exactly the regime where the naive formula
collapses. It compares your result to an independently computed reference
(mean first, then a second pass over the centered data) using

$$
\mathrm{rel\_err} = \frac{|\hat{\sigma}^2 - \sigma^2_{\text{ref}}|}{|\sigma^2_{\text{ref}}| + 10^{-12}} .
$$

The gate requires $\mathrm{rel\_err} \le 10^{-6}$. The naive one-pass formula
applied to this fixture is off by many orders of magnitude (and can return a
negative "variance"), so it fails this gate by a wide margin — only an
implementation that avoids the catastrophic cancellation passes.
