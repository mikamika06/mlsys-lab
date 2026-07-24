## Context

In many language models the final linear projection that maps hidden states to vocabulary logits is *tied* to the embedding matrix used for input tokens.  
Let $\mathbf{E}\in\mathbb{R}^{V\times D}$ denote the embedding matrix, where $V$ is the vocabulary size and $D$ the dimensionality of the hidden state.  
The tied‑head logits are computed as

$$
\mathbf{L} = \mathbf{E}\,\mathbf{E}^\top .
$$

If a separate head matrix $\mathbf{H}$ is used instead, the logits become $\mathbf{E}\,\mathbf{H}^\top$, which generally yields different values and can lead to training instability or degraded performance.

## Task

Implement the function `tied_head_logits` that takes an embedding matrix and returns the tied‑head logits:

```python
def tied_head_logits(embedding_matrix: np.ndarray) -> np.ndarray:
    ...
```

The returned array must have shape `(V, V)` and be of type `float64`.  No Python loops are allowed; use NumPy vectorized operations only.

## Example

```python
import numpy as np
E = np.array([[1.0, 0.0],
              [0.0, 1.0]])
L = tied_head_logits(E)
print(L)
# [[1. 0.]
#  [0. 1.]]
```

Here the embeddings form an identity matrix, so the logits are also the identity.

## What the gate checks

The grader computes a reference implementation using `E @ E.T` and compares your output with it via the metric

$$
\mathrm{max\_abs\_err} = \max_{i,j}\bigl|\,L_{\text{cand}}(i,j)-L_{\text{ref}}(i,j)\bigr|.
$$

Your solution must satisfy $\mathrm{max\_abs\_err}\le 10^{-5}$.
