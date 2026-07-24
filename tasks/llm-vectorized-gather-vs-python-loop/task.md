## Context

In language models the embedding lookup is a core operation. For a batch of token indices $i_1,\dots,i_n$ we need to retrieve their corresponding vectors from an embedding matrix $E \in \mathbb{R}^{V\times D}$, where $V$ is vocabulary size and $D$ the hidden dimension. The naive implementation iterates over the indices in Python:

$$
\texttt{for }k=1\ldots n:\quad h_k = E[i_k].
$$

This incurs a Python loop and many small array copies. NumPy supports fancy indexing, which performs the same gather in C and returns all rows at once:

$$
H = E[\mathbf{i}],
$$

where $\mathbf{i}$ is the 1‑D index array. The vectorized form is not only faster but also more memory efficient.

## Task

Implement `gather_embeddings(indices, embedding_matrix)` that takes a 1‑D NumPy array of integer indices and a 2‑D NumPy array representing the embedding matrix, and returns a 2‑D array containing the embeddings for each index. The result must be of type `float64`. Do not use explicit Python loops; rely on NumPy’s advanced indexing.

## Example

```python
import numpy as np
E = np.arange(20).reshape(5,4)   # 5 tokens, dim 4
idx = np.array([3,1,4])
emb = gather_embeddings(idx, E)
print(emb)
# [[12 13 14 15]
#  [ 4  5  6  7]
#  [16 17 18 19]]
```

## What the gate checks

Two metrics are evaluated:

* `max_abs_err`: The maximum absolute difference between your output and a reference implementation. It must be ≤ $10^{-7}$.
* `op_count`: The number of Python line events executed inside your function, counted with `sys.settrace`. It must be ≤ 50.

A solution that uses a Python loop will exceed the line‑event limit and fail the gate, while a fully vectorized implementation will pass both metrics.
