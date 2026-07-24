## Context

Attention computes weighted combinations of value vectors using query-key similarity.

For query matrix $Q \in \mathbb{R}^{N \times d}$, key matrix $K \in \mathbb{R}^{N \times d}$,
and value matrix $V \in \mathbb{R}^{N \times d_v}$, dense attention is

$$
O = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

The dense method creates a score matrix $S \in \mathbb{R}^{N \times N}$:

$$
S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} .
$$

FlashAttention avoids storing $S$ by processing query rows and key rows in tiles. A query
tile of size $B_r$ is combined with key tiles of size $B_c$. Each score tile is consumed
immediately to update a running softmax and output accumulator.

For numerical stability, the softmax can be maintained with running maximum $m$ and
normalizer $\ell$. When a new score tile changes the maximum to $m_{\mathrm{new}}$,

$$
m_{\mathrm{new}} = \max(m, m_{\mathrm{tile}})
$$

and

$$
\ell_{\mathrm{new}}
=
e^{m-m_{\mathrm{new}}}\ell
+
\sum_j e^{s_j-m_{\mathrm{new}}}.
$$

The output accumulator is updated using the same rescaling factor so that the final
result is equivalent to dense attention.

## Task

Implement `flash_attention_forward(Q, K, V, Br, Bc)`:

```python
def flash_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    Br: int,
    Bc: int
) -> np.ndarray:
    ...
```

Return the attention output with shape $(N, d_v)$.

Requirements:

- Use an outer loop over query tiles and an inner loop over key tiles.
- Do not construct a full $N \times N$ attention score matrix.
- Use NumPy operations within each tile.
- Match dense NumPy attention within the gate tolerance.

## Example

```python
import numpy as np

Q = np.array([[1., 0.], [0., 1.]])
K = np.array([[1., 0.], [0., 1.]])
V = np.array([[2., 3.], [4., 5.]])

O = flash_attention_forward(Q, K, V, 1, 1)
```

The result is approximately

```text
[[2.53788284, 3.53788284],
 [3.46211716, 4.46211716]]
```

## What the gate checks

The numeric gate builds a dense NumPy attention result as the oracle and checks

$$
\max_i |O_i-\hat{O}_i| \le 10^{-5}.
$$

The execution gate verifies that the implementation enters both tile loops.

The memory gate monitors NumPy allocations and rejects implementations that create a
live array with shape $(N,N)$. A dense score matrix implementation therefore fails even
if its output is numerically correct.
