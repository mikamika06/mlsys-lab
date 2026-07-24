## Context

GPTQ-style weight quantization uses an approximate Hessian matrix $H$ to decide how
errors from quantizing one weight column should be compensated in later columns.
For a weight matrix $W \in \mathbb{R}^{m \times n}$, the columns are the input
features being quantized.

The activation-order heuristic (`desc_act`) sorts columns by descending Hessian
diagonal values:

$$
p = \operatorname{argsort}(-\operatorname{diag}(H)).
$$

The permutation moves high-activation columns first. After reordering the columns,
the quantization loop processes columns in the order $p$. The final result must be
mapped back to the original column layout using the inverse permutation.

For this task, the GPTQ update is simplified to the following deterministic
algorithm. Let $Q_j$ be the symmetric per-column uniform 4-bit quantization of the
current column. After quantizing column $j$, the remaining unprocessed columns are
updated by

$$
W_{:,k} \leftarrow W_{:,k} -
W_{:,j}^{\mathrm{err}}
\frac{H^{-1}_{jk}}{H^{-1}_{jj}},
$$

where $W_{:,j}^{\mathrm{err}} = W_{:,j} - Q_j$.

This uses the inverse Hessian to compensate future columns before they are
quantized.

## Task

Implement:

```python
def gptq_act_order(W: np.ndarray, H: np.ndarray):
    ...
```

The input `W` is a float64 matrix of shape $(m,n)$ and `H` is a positive definite
float64 matrix of shape $(n,n)$.

Return a tuple:

```python
(permutation, W_hat)
```

where:

- `permutation` is an integer NumPy array containing the column order used by
  the algorithm.
- `W_hat` is the final quantized matrix with columns restored to the original
  order.

Use activation ordering by sorting columns with decreasing values of
$\operatorname{diag}(H)$. The quantization and error compensation procedure must
follow the formula in the context section.

## Example

```python
import numpy as np

W = np.array([[0.2, 1.1, -0.4],
              [0.7, 0.3,  0.9]], dtype=np.float64)

H = np.array([[5.0, 0.1, 0.0],
              [0.1, 2.0, 0.2],
              [0.0, 0.2, 1.0]], dtype=np.float64)

perm, W_hat = gptq_act_order(W, H)

# perm starts with column 0 because H[0,0] is largest.
# W_hat has the same column layout as the original W.
```

## What the gate checks

The gate computes an independent NumPy oracle implementation of the activation
ordered GPTQ procedure. It checks that the returned permutation exactly matches
the oracle and that the returned matrix has relative error

$$
\mathrm{rel\_err} =
\frac{\lVert W_{\mathrm{hat}}-W_{\mathrm{oracle}}\rVert_2}
{\lVert W_{\mathrm{oracle}}\rVert_2 + 10^{-12}}
$$

below the required threshold.

The oracle also computes the natural column order internally to ensure that the
activation-ordered result is a genuine improvement for the generated cases.
