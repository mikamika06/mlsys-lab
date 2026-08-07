## Context

Scaled dot-product attention computes

$$\operatorname{Attn}(Q, K, V)
  = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V,$$

where $Q, K \in \mathbb{R}^{n \times d_k}$ and $V \in \mathbb{R}^{n \times d_v}$.
A naïve implementation materialises the full $n \times n$ score matrix
$S = QK^\top / \sqrt{d_k}$, applies softmax row-wise, then multiplies by $V$.
This requires $O(n^2)$ memory and $O(n^2 d_k)$ floating-point operations,
both prohibitive when the sequence length $n$ is large.

Flash attention eliminates the $O(n^2)$ matrix by processing $Q$, $K$, $V$
in blocks (tiles) along the sequence dimension.  For each query block
$Q_i$ (rows $[i \cdot b_r,\;(i{+}1) \cdot b_r)$), we iterate over
key/value block pairs $(K_j, V_j)$ and maintain a **running** online softmax
with three scalars per query row:

$$m^{(j)} = \max\!\bigl(m^{(j-1)},\;
              \operatorname{rowmax}(S_{ij})\bigr),$$

$$\ell^{(j)} = \ell^{(j-1)} \cdot e^{\,m^{(j-1)} - m^{(j)}}
             + \operatorname{rowsum}\!\bigl(
                 e^{\,S_{ij} - m^{(j)}}\bigr),$$

$$O_i^{(j)} = O_i^{(j-1)} \cdot e^{\,m^{(j-1)} - m^{(j)}}
            + e^{\,S_{ij} - m^{(j)}} \, V_j,$$

where $S_{ij} = Q_i K_j^\top / \sqrt{d_k}$ is the **local**
$b_r \times b_c$ score block, never the full matrix.  After every key block
the rescaling factor $e^{\,m^{(j-1)}-m^{(j)}}$ corrects the previously
accumulated softmax numerator for the new global maximum.  When all key
blocks have been consumed the accumulator is normalised:

$$O_i = O_i^{(J)} \,\big/ \bigl(\ell^{(J)}\bigr)^\top.$$

The total working memory is $O(n \cdot d)$ rather than $O(n^2)$.

## Task

Implement

```python
def tiled_flash_attention_forward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int=64) -> list[list[float]]:
    ...
```

**Parameters:**

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `Q` | list[float] float64 | $(n, d_k)$ | Queries |
| `K` | list[float] float64 | $(n, d_k)$ | Keys |
| `V` | list[float] float64 | $(n, d_v)$ | Values |
| `block_size` | `int` | — | Tile width $b_r = b_c$ for both query and key/value blocks |

**Returns:**

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `O` | list[float] float64 | $(n, d_v)$ | Attention output |

Implement the block-tiled online-softmax flash-attention forward pass
described above.  You must **not** materialise the full $n \times n$ score
matrix at any point.  When $n$ is not evenly divisible by `block_size`, the
last block should cover the remaining rows.

## Example

```python

rng = random.Random(0)
Q = rng.randn(64, 16)
K = rng.randn(64, 16)
V = rng.randn(64, 8)

O = tiled_flash_attention_forward(Q, K, V, block_size=32)
assert O.shape == (64, 8)
```

## What the gate checks

The gate computes `max_abs_err`, the maximum elementwise absolute difference
between the learner's output and a float64 reference obtained by
materialising the full $n \times n$ softmax inside the grader.  Five test
cases with varying $(n, d_k, d_v, \texttt{block\_size})$ are evaluated,
including cases where $n$ is not divisible by `block_size`, where
`block_size` $\ge n$, and where $d_k \ne d_v$.  The gate passes when
$\texttt{max\_abs\_err} < 10^{-4}$.
