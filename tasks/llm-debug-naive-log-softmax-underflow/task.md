## Context

The softmax function converts a vector of real scores into a probability
distribution.  For an input vector $x \in \mathbb{R}^n$:

$$\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$$

Many loss functions (cross-entropy, KL divergence) require the *logarithm* of
softmax rather than softmax itself.  A naive implementation computes
$\log\!\bigl(\operatorname{softmax}(x)\bigr)$ by evaluating the softmax first
and then taking the element-wise logarithm:

$$\hat{y}_i = \ln\!\left(\frac{e^{x_i}}{\sum_j e^{x_j}}\right)$$

This two-step approach is numerically unstable.  When $x_i$ is a large
negative number, $e^{x_i}$ underflows to $0.0$ in IEEE 754 double-precision
arithmetic, and $\ln(0.0)$ produces $-\infty$.  Even when the *true* value of
$\log \operatorname{softmax}(x)_i$ is a perfectly finite number (e.g.\ $-3.7$),
the naive code returns $-\infty$.

The standard fix is the **log-sum-exp trick**.  Subtract the maximum element
$m = \max_j x_j$ before exponentiating:

$$\operatorname{log\_softmax}(x)_i = x_i - m - \ln\!\left(\sum_{j=1}^{n} e^{x_j - m}\right)$$

Because every exponent argument $x_j - m \le 0$, the largest value of
$e^{x_j - m}$ is $e^0 = 1$, so no overflow can occur, and the sum is always
$\ge 1$ so the logarithm is always finite.  The subtracted constant $m$
cancels algebraically, so the result is mathematically identical.

## Task

The file `starter.py` contains a **buggy** implementation of `log_softmax`.
It uses the naive two-step approach described above.  Your job is to fix the
function so that it returns correct, finite values for all inputs, including
large negative values.

```python
def log_softmax(x):
    """Compute log(softmax(x)) along the last axis.

    x : list of shape (..., n)
    Returns: list of same shape, with log-softmax applied along the last axis.
    """
    ...
```

The input may be 1-D (a single vector) or 2-D (a batch of vectors). Always return a `float64` list of the same shape.

## Example

```python
x = [-1000.0, -1001.0, -1002.0]
y = log_softmax(x)
# Correct: approximately [-0.4076, -1.4076, -2.4076]
# Bug:     [-inf, -inf, -inf]
```

With a normal-scale input the naive code happens to work:

```python
x = [1.0, 2.0, 3.0]
y = log_softmax(x)
# Both naive and fixed give approximately [-2.4076, -1.4076, -0.4076]
```

## What the gate checks

Two gates.

**`max_abs_err`** — The maximum element-wise absolute difference between your output and a reference log-softmax computed with the log-sum-exp trick must be below $10^{-6}$. Five test cases are used, including inputs where every element is around $-1000$ (complete naive underflow), mixed inputs where some elements underflow and others do not, and batched 2-D inputs.

**`all_finite`** — Every element of the output across all test cases must be finite (`math.isfinite`). The buggy starter returns $-\infty$ on the large- negative inputs, so it fails this gate immediately.
