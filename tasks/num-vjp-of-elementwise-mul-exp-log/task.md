## Context

The elementwise product of two vectors $x, y \in \mathbb{R}^n$ is defined by $(x\!\cdot\!y)_i = x_i\,y_i$.  
Applying the exponential and logarithm elementwise gives

$$h(x,y) \;=\;\log\bigl(\exp(x\!\cdot\!y)\bigr)
   \;=\;x\!\cdot\!y,$$

because $\log(\exp(z)) = z$ for every real $z$.  
The Jacobian of this composite operation with respect to the first argument is
$\partial h_i/\partial x_j = y_i\,\delta_{ij}$, and similarly for $y$.

In automatic differentiation a **vector‑Jacobian product** (VJP) takes an upstream
gradient vector $v \in \mathbb{R}^n$ (the gradient of some scalar loss with respect
to the output of $h$) and returns the gradients with respect to each input:
$$\frac{\partial L}{\partial x}
   = v^\top\,\frac{\partial h}{\partial x},
\qquad
\frac{\partial L}{\partial y}
   = v^\top\,\frac{\partial h}{\partial y}.$$
For the operation above this simplifies to

$$\boxed{
  \frac{\partial L}{\partial x} = v \,\odot\, y,
  \qquad
  \frac{\partial L}{\partial y} = v \,\odot\, x
}$$

where $\odot$ denotes elementwise multiplication.

## Task

Implement the function `vjp_mul_exp_log` that computes these gradients:

```python
def vjp_mul_exp_log(x: list[float], y: list[float], upstream: list[float]) -> tuple[list[float], list[float]]:
    """
    Compute the vector-Jacobian product of h(x,y)=log(exp(x*y))
    with respect to x and y.

    Parameters
    ----------
    x : 1-D array of float64
        First input vector.
    y : 1-D array of float64
        Second input vector, same shape as `x`.
    upstream : 1-D array of float64
        Upstream gradient (same shape as the output of h).

    Returns
    -------
    grad_x, grad_y : tuple of arrays
        Gradients with respect to x and y, each a float64 array of the same shape.
    """
```

All inputs are guaranteed to be list of floats of equal length,
and all outputs must also be `float64` arrays.

## Example

```python
x = [1.0, 2.0, -3.0]
y = [4.0, -5.0, 6.0]
upstream = [0.1, 0.2, 0.3]

grad_x, grad_y = vjp_mul_exp_log(x, y, upstream)
print(grad_x)   # [0.4, -1.0, -1.8]
print(grad_y)   # [0.1, -1.0, 1.8]
```

## What the gate checks

The grader computes a reference VJP by central finite differences of the
composite function $h(x,y)=\log(\exp(x*y))$.  
It then compares your implementation against this reference using the
`max_abs_err` scorer from `arena.scorers`.  Your solution must achieve
$\mathrm{max\_abs\_err} \le 10^{-5}$.

A correct implementation will use only Python vectorised operations and
return arrays of type `float64`.
