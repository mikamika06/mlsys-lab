## Context

Scaled dot-product attention computes a weighted combination of value vectors. Given query
matrix $Q \in \mathbb{R}^{n \times d}$, key matrix $K \in \mathbb{R}^{m \times d}$, and value
matrix $V \in \mathbb{R}^{m \times h}$, the attention output is

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}} + B\right)V,
$$

where $B$ is an optional attention bias matrix. The softmax is applied row-wise:

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} .
$$

Production libraries provide multiple implementations of scaled dot-product attention. The
memory-efficient implementation avoids materializing the full attention matrix, while the
math backend follows the direct matrix formulation. For the same inputs, both should compute
the same mathematical result up to floating-point error.

## Task

Implement `compare_sdpa_backends(Q, K, V, bias)`:

```python
def compare_sdpa_backends(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    bias: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

Return a pair `(mem_efficient_output, math_output)`.

The first returned array represents the xformers-style memory-efficient path. The second
represents the straightforward SDPA math backend. Both outputs must have shape
$(n, h)$ and dtype `float64`.

The function must support an optional bias matrix with shape $(n, m)$. When `bias` is
`None`, no bias is added. Use numerically stable softmax computation.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0, 3.0], [4.0, 5.0]])
bias = None

mem, math = compare_sdpa_backends(Q, K, V, bias)

# mem and math contain the same attention result up to floating point error
```

## What the gate checks

The gate computes a NumPy fp64 attention oracle using the same bias and compares both
returned backend results against it. It also checks that the two returned arrays agree.

The reported metric is

$$
\max\left(
\lVert M - O \rVert_\infty,
\lVert S - O \rVert_\infty,
\lVert M - S \rVert_\infty
\right),
$$

where $M$ is the memory-efficient result, $S$ is the math-backend result, and $O$ is the
oracle result. The value must be at most $10^{-5}$.
