## Context

Layer normalization transforms each row of an input matrix by removing its mean and
normalizing its variance. For a row vector $x \in \mathbb{R}^d$:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i,
$$

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2,
$$

and the normalized output is

$$
y_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}.
$$

In reverse mode automatic differentiation, the vector-Jacobian product (VJP)
computes how an incoming gradient $g$ on $y$ contributes to the gradient on $x$.

For layer normalization without learned scale and bias, the backward formula for
one row can be written as:

$$
\frac{\partial L}{\partial x_i}
=
\frac{1}{\sqrt{\sigma^2+\epsilon}}
\left(
g_i
-
\mathrm{mean}(g)
-
\hat{x}_i \mathrm{mean}(g\hat{x})
\right),
$$

where

$$
\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}.
$$

The implementation should apply this chain rule directly with Python operations.

## Task

Implement `layernorm_vjp(x, grad_y, eps)`:

```python
def layernorm_vjp(x: list[list[float]], grad_y: list[list[float]], eps: float=1e-05) -> list[list[float]]:
    ...
```

The inputs `x` and `grad_y` are list of the same shape $(n,d)$. Each row
is normalized independently. Return the gradient with respect to `x` with the
same shape and `float64` dtype.

Do not use automatic differentiation libraries. Use Python operations only.

## Example

```python

x = [[1.0, 2.0, 3.0]]
grad_y = [[1.0 for _ in row] for row in x]

dx = layernorm_vjp(x, grad_y)
# dx is approximately [[0.0, 0.0, 0.0]]
```

## What the gate checks

The gate computes a numerical gradient oracle using central finite differences on
the layer normalization forward computation. The returned VJP is compared against
the oracle using maximum absolute error:

$$
\mathrm{max\_abs\_err} = \max_i |g_i^{candidate}-g_i^{oracle}|.
$$

The error must be less than $10^{-4}$.
