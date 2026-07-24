## Context

A feed-forward network block in a transformer can use a gated activation instead of a
single hidden layer. SwiGLU splits the input projection into two paths. One path is
the gate and the other is the value projection.

For an input matrix $X \in \mathbb{R}^{n \times d}$, gate projection
$W_g \in \mathbb{R}^{d \times h}$, value projection
$W_u \in \mathbb{R}^{d \times h}$, and output projection
$W_d \in \mathbb{R}^{h \times d}$, the SwiGLU computation is

$$
H = \operatorname{silu}(XW_g) \odot (XW_u),
$$

$$
Y = HW_d.
$$

The SiLU activation is defined as

$$
\operatorname{silu}(z) = z \sigma(z) =
\frac{z}{1 + e^{-z}} .
$$

The element-wise product $\odot$ lets the gate path control which hidden features
are passed through the feed-forward network.

## Task

Implement `swiglu_ffn`:

```python
def swiglu_ffn(
    x: np.ndarray,
    gate_w: np.ndarray,
    up_w: np.ndarray,
    down_w: np.ndarray,
) -> np.ndarray:
    ...
```

The arguments are NumPy arrays with shapes:

- `x`: $(n, d)$
- `gate_w`: $(d, h)$
- `up_w`: $(d, h)$
- `down_w`: $(h, d)$

Return the output array $Y$ with shape $(n, d)$.

The implementation must compute the SwiGLU feed-forward operation using NumPy
operations. The result must be `float64`.

## Example

```python
import numpy as np

x = np.array([[1.0, -1.0]])
gate_w = np.array([[0.5, 0.2], [0.1, -0.3]])
up_w = np.array([[0.4, -0.2], [0.6, 0.1]])
down_w = np.array([[1.0, 0.5], [-0.5, 0.7]])

y = swiglu_ffn(x, gate_w, up_w, down_w)
```

## What the gate checks

The gate computes the expected output using an independent NumPy oracle that
implements the SwiGLU equations. It compares the submitted implementation using
the maximum absolute error

$$
\max_i |Y_i - \hat{Y}_i|.
$$

The value must be below $10^{-5}$. This catches incorrect gating order, missing
the SiLU activation, and incorrect projection order.
