## Context

FlashAttention-style backward avoids materializing the full attention matrix. The forward pass computes

$$
S = \frac{QK^\top}{\sqrt{d}},
$$

then applies an optional causal mask and softmax:

$$
P_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}.
$$

Instead of storing all of $P$, the forward pass stores the row logsumexp values

$$
L_i = \log \sum_j e^{S_{ij}} .
$$

During backward, the attention probabilities are recomputed as

$$
P_{ij} = e^{S_{ij} - L_i}.
$$

Given an output gradient $dO$, the attention backward equations are

$$
dV = P^\top dO,
$$

$$
dP = dO V^\top,
$$

$$
dS_{ij} = P_{ij}\left(dP_{ij} - \sum_k dP_{ik}P_{ik}\right),
$$

and

$$
dQ = \frac{dS K}{\sqrt{d}}, \qquad dK = \frac{dS^\top Q}{\sqrt{d}}.
$$

A production implementation uses these equations while recomputing $P$ from $L$ instead of storing the complete attention matrix.

## Task

Implement `flash_backward_dq_dk_dv`:

```python
def flash_backward_dq_dk_dv(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    dO: np.ndarray,
    logsumexp: np.ndarray,
    causal: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

The inputs are float64 arrays with shapes $(n,d)$, $(n,d)$, $(n,m)$, $(n,m)$, and $(n,)$. The function must return $(dQ, dK, dV)$ with shapes matching $Q$, $K$, and $V$.

The `logsumexp` input is produced by the forward pass. Recompute the attention probabilities from $Q$, $K$, and this vector. Do not use a stored full attention matrix.

When `causal` is true, positions where $j > i$ are masked out before softmax.

## Example

```python
import numpy as np

Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[2.0], [3.0]])
dO = np.ones((2, 1))

S = Q @ K.T / np.sqrt(2)
L = np.log(np.exp(S).sum(axis=1))

dQ, dK, dV = flash_backward_dq_dk_dv(Q, K, V, dO, L, False)
```

## What the gate checks

The gate builds random small attention problems with both causal and non-causal modes. It computes the reference gradients using a central finite-difference oracle over the scalar attention loss.

The returned gradients are compared using the relative error metric

$$
\mathrm{rel\_err} =
\frac{\lVert x_{\mathrm{candidate}}-x_{\mathrm{oracle}}\rVert_2}
{\lVert x_{\mathrm{oracle}}\rVert_2 + 10^{-12}} .
$$

The combined gradient error across $dQ$, $dK$, and $dV$ must satisfy the required threshold.
