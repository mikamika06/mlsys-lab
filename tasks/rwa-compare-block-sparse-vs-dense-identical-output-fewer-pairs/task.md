## Context

Attention between query vectors $Q$, key vectors $K$, and value vectors $V$ computes a weighted combination of values. For a query row $i$, the dense masked attention output is

$$
O_i = \sum_j P_{ij} V_j,
$$

where

$$
P = \mathrm{softmax}(S), \qquad S = \frac{QK^\top}{\sqrt{d}}.
$$

A boolean attention mask removes pairs that are not allowed. A masked pair contributes exactly zero because its probability is forced to zero before the softmax normalization.

A block-sparse implementation divides the $n \times n$ attention matrix into square blocks. If a whole block of the mask contains only `False` entries, that block cannot contribute to the output. Skipping it reduces the number of query-key pairs processed while preserving the dense masked result.

For a block size $b$, each block contains $b^2$ possible query-key pairs. If $e$ blocks are empty, the sparse kernel avoids

$$
e b^2
$$

pairs compared with the dense computation.

## Task

Implement `block_sparse_attention(Q, K, V, mask, block_size)`:

```python
def block_sparse_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray,
    block_size: int
) -> tuple[np.ndarray, int]:
    ...
```

The inputs are:

- `Q` with shape $(n, d)$ containing query vectors.
- `K` with shape $(n, d)$ containing key vectors.
- `V` with shape $(n, m)$ containing value vectors.
- `mask` with shape $(n, n)$ containing boolean allowed attention pairs.
- `block_size` dividing $n$.

Return:

1. The dense-equivalent masked attention output as a `float64` NumPy array of shape $(n, m)$.
2. The number of query-key pairs represented by processed non-empty blocks.

The implementation must skip blocks where the corresponding mask region is entirely `False`. The softmax normalization must still be computed over all allowed keys for each query row, even when the keys are processed block by block.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0], [4.0]])
mask = np.array([[True, False], [False, False]])

out, pairs = block_sparse_attention(Q, K, V, mask, 1)

# out matches dense masked attention
# pairs == 1
```

## What the gate checks

The gate computes a NumPy float64 dense masked-attention oracle and compares the returned output using `max_abs_err`.

The `attended_pairs` value must equal the number of pairs inside non-empty mask blocks:

$$
n^2 - (\texttt{empty\_blocks} \times \texttt{block\_size}^2).
$$

A solution that computes the correct dense output but always reports $n^2$ pairs fails the pair-count check.
