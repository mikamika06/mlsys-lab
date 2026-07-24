## Context

Attention over a sequence of query vectors $Q$, key vectors $K$, and value vectors $V$ is normally computed by forming the complete score matrix:

$$
S = \frac{QK^\top}{\sqrt{d}},
$$

then applying a row-wise softmax and multiplying by the values:

$$
O = \mathrm{softmax}(S)V .
$$

For a long context, systems often split the sequence across multiple devices. Ring attention simulates this by distributing blocks of keys and values across ranks. Each rank rotates key/value blocks around a ring and incrementally merges the partial attention results.

The online softmax merge keeps a running maximum $m$, normalization value $l$, and output accumulator $o$:

$$
m_{\text{new}} = \max(m, m_{\text{block}}),
$$

$$
l_{\text{new}} =
e^{m-m_{\text{new}}}l +
\sum_j e^{s_j-m_{\text{new}}},
$$

$$
o_{\text{new}} =
\frac{
e^{m-m_{\text{new}}}l\,o +
\sum_j e^{s_j-m_{\text{new}}}v_j
}{
l_{\text{new}}}.
$$

After all key/value blocks have visited each rank, the result should equal ordinary full attention.

## Task

Implement `ring_attention(Q, K, V, ranks)`.

The inputs are NumPy arrays:

```python
def ring_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    ranks: int,
) -> np.ndarray:
    ...
```

`Q`, `K`, and `V` have shapes $(n, d)$, $(n, d)$, and $(n, d_v)$.
The sequence is split into `ranks` contiguous key/value blocks. Simulate the ring by rotating these blocks and using online softmax merging. Do not call a full attention helper that directly computes all $QK^\top$ scores at once.

Return the final output matrix with shape $(n, d_v)$ and dtype `float64`.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0, 3.0], [4.0, 5.0]])

O = ring_attention(Q, K, V, 2)
```

The result matches:

$$
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{2}}\right)V .
$$

## What the gate checks

The gate computes a NumPy full-attention oracle independently:

$$
O_{\mathrm{ref}} =
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V .
$$

The returned matrix is compared using the maximum absolute element error:

$$
\max_i |O_i - O_{\mathrm{ref},i}|.
$$

The error must be below $10^{-5}$. A solution that only attends to the local key/value block will fail because it ignores the rotated ring blocks.
