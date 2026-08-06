## Context

The softmax function converts a vector of logits $x \in \mathbb{R}^n$ into a probability vector $s$:

$$
s_i = \frac{e^{x_i}}{\sum_j e^{x_j}} .
$$

In reverse-mode automatic differentiation, the vector-Jacobian product (VJP) applies the transpose of the Jacobian to an incoming gradient $g$.

For softmax, the VJP simplifies to:

$$
\frac{\partial L}{\partial x_i}
=
s_i \left(g_i - \sum_j g_j s_j\right).
$$

This avoids constructing the full $n \times n$ Jacobian matrix and directly computes the gradient needed to propagate through the operation.

## Task

Implement `softmax_vjp(x, g)`:

```python
def softmax_vjp(x: list[float], g: list[float]) -> list[float]:
    ...
```

The function receives two list of floats with the same shape. `x` contains logits and `g` is the gradient arriving from the output of softmax. Return the gradient with respect to `x`.

Requirements:
- Use Python operations.
- Return a `float64` list.
- Handle arbitrary finite input values.

## Example

```python

x = [1.0, 2.0, 3.0]
g = [0.5, -1.0, 2.0]

dx = softmax_vjp(x, g)
```

The output is the gradient that should be passed to the operation that produced `x`.

## What the gate checks

The gate computes a Python softmax reference and compares the returned VJP against it. The maximum absolute error

$$
\max_i |\hat{d x}_i - d x_i|
$$

must be less than $10^{-5}$. Implementations that use an incorrect chain rule or omit the weighted sum term will fail.
