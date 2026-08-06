## Context

The softmax function maps a vector of real numbers $z \in \mathbb{R}^k$ to a probability distribution over its components:

$$\operatorname{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{k}\exp(z_j)}.$$

When the entries of $z$ are large in magnitude, computing $\exp(z_i)$ can overflow or underflow, producing `inf` or `0`. A standard numerical fix is to subtract the maximum entry before exponentiation:

$$\operatorname{softmax}(z_i) = \frac{\exp(z_i - m)}{\sum_{j=1}^{k}\exp(z_j - m)}, \qquad
m = \max_{j} z_j.$$

This transformation preserves the ratios of the exponentials and thus the resulting probabilities, while keeping all intermediate values in a safe range.

## Task

Implement a function `softmax(logits)` that:

* accepts a list `logits` of arbitrary shape,
* computes the softmax along the last axis using the max‑subtraction trick,
* returns an array of type `float64` with the same shape as the input.

The implementation must use only vectorized Python operations; explicit Python loops are disallowed.

## Example

```python
logits = [1000.0, 1001.0, -1000.0]
probs = softmax(logits)
print(probs)  # [0.2689414213699951, 0.7310585786300049, 0.0]
```

The large positive values do not cause overflow because the maximum is subtracted before exponentiation.

## What the gate checks

The grader computes a reference softmax using Python’s stable implementation and compares your output with it via the metric `max_abs_err`. Your result must satisfy

$$\mathrm{max\_abs\_err} \le 10^{-9}.$$

Any occurrence of `NaN` or `inf` in the returned array will cause the gate to fail.
