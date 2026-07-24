## Context

FlashAttention avoids storing the full attention score matrix during the forward pass. Instead of saving
the softmax probability matrix $P$, it stores the row-wise logsumexp values:

$$
\mathrm{lse}_i = \log \sum_j \exp(S_{ij}),
$$

where the scaled attention scores are

$$
S = \frac{QK^\top}{\sqrt{d}} .
$$

During backward, the probability matrix can be reconstructed from $Q$, $K$, and the stored
$\mathrm{lse}$:

$$
P_{ij} = \exp(S_{ij} - \mathrm{lse}_i).
$$

The gradients can then be computed without requiring a saved score matrix. For an upstream gradient
$dO$, the value gradient is

$$
dV = P^\top dO .
$$

The softmax backward pass uses

$$
dS_{ij} = P_{ij}\left(dP_{ij} - \sum_k dP_{ik}P_{ik}\right),
$$

where

$$
dP = dO V^\top .
$$

The query and key gradients are

$$
dQ = \frac{dS K}{\sqrt{d}}, \qquad
dK = \frac{dS^\top Q}{\sqrt{d}} .
$$

## Task

Implement `flash_backward`:

```python
def flash_backward(Q, K, V, dO, lse):
    ...
```

Inputs are NumPy arrays:

- `Q` has shape $(n, d)$.
- `K` has shape $(n, d)$.
- `V` has shape $(n, dv)$.
- `dO` has shape $(n, dv)$ and is the gradient arriving from the output.
- `lse` has shape $(n,)$ and contains the row logsumexp values computed during forward.

Return a tuple `(dQ, dK, dV)` with the same shapes as `Q`, `K`, and `V`.

Reconstruct the softmax probabilities from the stored logsumexp values. Do not use a precomputed attention score or probability matrix passed by the caller.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0], [3.0]])
dO = np.ones((2, 1))

S = Q @ K.T / np.sqrt(2)
lse = np.log(np.exp(S).sum(axis=1))

dQ, dK, dV = flash_backward(Q, K, V, dO, lse)
```

## What the gate checks

The gate computes the reference backward pass itself using the FlashAttention equations and compares all
three returned gradients. The reported metric is the maximum absolute error:

$$
\max(|dQ-\hat{dQ}|, |dK-\hat{dK}|, |dV-\hat{dV}|).
$$

The submission passes when this value is at most $10^{-5}$.
