## Context

In quantized neural network inference, an int8 General Matrix Multiply (GEMM) accelerates $Y = X W$ by representing operands as 8-bit integers. Each vector (row of $X$ or column of $W$) uses its own scale factor based on the absolute maximum value.

For a row $x \in \mathbb{R}^d$ of the activation matrix $X \in \mathbb{R}^{n \times d}$:

$$
s_x = \frac{\max_{j} |x_j|}{127}
$$

For a column $w \in \mathbb{R}^d$ of the weight matrix $W \in \mathbb{R}^{d \times m}$:

$$
s_w = \frac{\max_{i} |w_i|}{127}
$$

## Task

Implement `compute_int8_scales(X, W)` that returns two 1-D arrays: per-row activation scales `scale_x` (length $n$) and per-column weight scales `scale_w` (length $m$).

```python
def compute_int8_scales(X: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ...
```

- `X` has shape $(n, d)$, `W` has shape $(d, m)$.
- Scale for each row of $X$: $\text{scale\_x}_i = \max_j |X_{ij}| / 127$.
- Scale for each column of $W$: $\text{scale\_w}_j = \max_i |W_{ij}| / 127$.

## Example

```python
import numpy as np
X = np.array([[ 1.0, -2.0,  3.0], [-4.0,  5.0, -6.0]])
W = np.array([[ 0.5, -1.5], [-3.5,  4.5]])
sx, sw = compute_int8_scales(X, W)
# sx ≈ [0.02362, 0.04724],  sw ≈ [0.02756, 0.03543]
```

## What the gate checks

Two gates compare your output against an oracle:

- `rel_err_x`: relative L2 error of `scale_x` vs oracle, must be $\le 10^{-6}$.
- `rel_err_w`: relative L2 error of `scale_w` vs oracle, must be $\le 10^{-6}$.

Oracle computes $\max(|\cdot|, \text{axis=...}) / 127.0$ in float64.
