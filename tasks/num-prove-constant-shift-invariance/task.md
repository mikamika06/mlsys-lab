## Context

The softmax function is defined for a vector $z \in \mathbb{R}^n$ as

$$\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}.$$

A numerically stable implementation subtracts the maximum element before exponentiating:

$$\operatorname{softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^n e^{z_j - \max(z)}}.$$

The softmax is invariant to additive constant shifts. For any constant $c \in \mathbb{R}$:

$$\operatorname{softmax}(z - c\mathbf{1})_i = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{z_i}}{\sum_j e^{z_j}} = \operatorname{softmax}(z)_i.$$

The constant $c$ cancels between numerator and denominator. We verify this numerically by computing softmax with and without a shift and measuring the maximum absolute difference.

## Task

Implement `softmax_shift_invariant(logits, shift)`:

```python

def softmax_shift_invariant(logits, shift):
    """
    Returns the maximum absolute error between softmax(logits) and
    softmax(logits - shift), proving numerical invariance to constant shifts.
    """
    ...
```

The function takes:
- `logits`: a list of shape `(n,)` or `(batch, n)`
- `shift`: an array of the same shape (or broadcastable) representing the constant to subtract

Compute numerically stable softmax of both `logits` and `logits - shift`. Return the maximum absolute difference between these two results.

## Example

```python
logits = [1.0, 2.0, 3.0]
shift = [100.0, 100.0, 100.0]
error = softmax_shift_invariant(logits, shift)
# error < 1e-12
```

## What the gate checks

The gate verifies that `max_abs_err` between the user's result and the Python reference result is less than $10^{-7}$. The reference computes both softmax values using stable max-subtraction. A correct implementation returns a value near zero; an incorrect one triggers a failure.
