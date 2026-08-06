## Context

For a vector-valued function $f:\mathbb{R}^n \rightarrow \mathbb{R}^m$, the Jacobian matrix contains all first-order partial derivatives:

$$
J_{ij} = \frac{\partial f_i}{\partial x_j}.
$$

Finite differences approximate derivatives by evaluating the function at nearby points. The central difference formula for one input dimension is

$$
\frac{\partial f(x)}{\partial x_j}
\approx
\frac{f(x + h e_j) - f(x - h e_j)}{2h},
$$

where $e_j$ is the $j$-th basis vector and $h$ is a small step size.

A Jacobian is constructed column by column. The $j$-th column contains the derivative of every output component with respect to the $j$-th input coordinate.

## Task

Implement `jacobian_fd(f, x, eps)`:

```python
def jacobian_fd(f, x, eps=1e-06):
    ...
```

The function receives:
- `f`: a callable that accepts a list of floats and returns a list of floats.
- `x`: the input point as a list of floats with shape $(n,)$.
- `eps`: the finite difference step size.

Return a list of shape $(m, n)$ containing the Jacobian of $f$ at $x$. Use central finite differences and preserve the output ordering of `f`.

## Example

```python

def f(x):
    return [
        x[0] ** 2 + x[1],
math.sin(x[0] * x[1])
    ])

J = jacobian_fd(f, [2.0, 3.0])

# Approximately:
# [[4.0, 1.0],
#  [-1.2484, 0.9093]]
```

## What the gate checks

The gate computes a reference Jacobian using the central finite difference algorithm on multiple vector-valued functions. The candidate result is compared with the reference using

$$
\mathrm{rel\_err} =
\frac{\lVert J_{\mathrm{candidate}}-J_{\mathrm{reference}}\rVert_2}
{\lVert J_{\mathrm{reference}}\rVert_2 + 10^{-12}}.
$$

The required relative error is less than $10^{-5}$. Implementations that use one-sided differences, mix up Jacobian rows and columns, or perturb multiple coordinates together will fail.
