## Context

The softmax function maps a vector of logits $z \in \mathbb{R}^d$ to a probability
distribution:

$$\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{d} e^{z_j}}.$$

A naive implementation computes $e^{z_i}$ directly. In IEEE 754 `float64`,
`math.exp(x)` overflows to `inf` once $x \gtrsim 709$. Even before the hard
overflow boundary, values like $e^{300} \approx 10^{130}$ are so large that
dividing two of them can lose significant digits. If every element overflows
to `inf`, the result is `nan` (inf / inf).

The standard fix exploits the translation invariance of softmax — shifting
all logits by the same constant leaves the output unchanged:

$$\sigma(z_i) = \frac{e^{z_i - m}}{\sum_{j=1}^{d} e^{z_j - m}},
\qquad m = \max_{j} z_j.$$

After subtracting the maximum, every exponent is $\le 0$, so
$e^{z_i - m} \in (0, 1]$ and overflow is impossible.

## Task

Implement `stable_softmax(logits)`:

```python
def stable_softmax(logits):
    """Compute softmax along the last axis, numerically stable."""
    ...
```

The input `logits` is a list of shape $(..., d)$ where $d \ge 1$.
Return an array of the same shape whose last axis sums to $1.0$ (within
floating-point tolerance). Every element must be a finite non-negative
number — no `inf`, no `nan`.

Work in `float64` internally. Do not call `scipy.special.softmax`.

## Example

```python
logits = [1000.0, 999.0, 0.0]
result = stable_softmax(logits)
# result ≈ [0.7311, 0.2689, 0.0]   (all finite, sums to 1)
# [math.exp(x) for x in [1000, 999, 0]] would give [inf, inf, 0.0]
```

## What the gate checks

The gate computes the maximum absolute error

$$\mathrm{max\_abs\_err} = \max_{i} |y_i - y_i^{\mathrm{ref}}|$$

where $y^{\mathrm{ref}}$ is the reference stable softmax computed by the
grader using the max-subtraction method. The test data includes rows with
logit values up to $+1000$, rows of uniform large values, rows spanning
$[-1000, +1000]$, and a random batch drawn from
$\mathrm{Uniform}(-1000, +1000)$. The gate passes when
$\mathrm{max\_abs\_err} < 10^{-7}$.

If the output contains any `inf` or `nan`, the gate reports
$\mathrm{max\_abs\_err} = \infty$ and fails.
A naive implementation that calls `math.exp` without subtracting the max
will produce `inf` / `nan` for rows containing logits above $\approx 709$.
