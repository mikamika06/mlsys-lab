## Context

Rotary Position Embedding (RoPE) injects absolute position information into the query and key tensors of a transformer attention layer by rotating each half‑dimensional slice of the vectors with a sinusoidal angle that depends on the token’s index.  
For a vector $x \in \mathbb{R}^d$ ($d$ even) we split it as
$$
x = (x_0, x_1, \dots , x_{d-2}, x_{d-1}) =
\bigl( x_{\text{even}},\, x_{\text{odd}} \bigr),
$$
where $x_{\text{even}}$ contains the even indices and $x_{\text{odd}}$ the odd ones.  
Given pre‑computed sine and cosine tables $\sin_{i,p}$, $\cos_{i,p}$ for position $i$ and pair index $p$, the rotated query is

$$
\tilde{x}_{\text{even},p} = x_{\text{even},p}\,\cos_{i,p}
- x_{\text{odd},p}\,\sin_{i,p},
\qquad
\tilde{x}_{\text{odd},p}  = x_{\text{even},p}\,\sin_{i,p}
+ x_{\text{odd},p}\,\cos_{i,p}.
$$

The standard attention score between positions $i$ and $j$ is then
$$
s_{ij} = \tilde{Q}_i \cdot \tilde{K}_j^\top .
$$

A naïve implementation materialises $\tilde{Q}$ and $\tilde{K}$ before the dot product, which doubles memory traffic.  The goal of this task is to fuse the rotation into the score computation so that no intermediate rotated tensors are created.

## Task

Implement a function with the following signature:

```python
def fused_rope_qk(
    Q: np.ndarray,
    K: np.ndarray,
    sin: np.ndarray,
    cos: np.ndarray
) -> np.ndarray:
```

* `Q` and `K` have shape `(batch, seq_len, dim)` where `dim` is even.  
* `sin` and `cos` each have shape `(seq_len, dim // 2)` and contain the pre‑computed sine and cosine values for every position and half‑dimension pair.  
* The function must return a tensor of shape `(batch, seq_len, seq_len)` containing the attention scores after RoPE has been fused into the dot product.  
* Use only NumPy vectorised operations; no explicit Python loops over tokens or batches.

## Example

```python
import numpy as np

# small toy example
batch, seq_len, dim = 1, 4, 6   # dim must be even
Q = np.random.randn(batch, seq_len, dim).astype(np.float32)
K = np.random.randn(batch, seq_len, dim).astype(np.float32)

# pre‑compute sin/cos for each position and half‑dimension pair
half = dim // 2
angles = np.linspace(0.1, 1.0, seq_len * half).reshape(seq_len, half)
sin = np.sin(angles).astype(np.float32)
cos = np.cos(angles).astype(np.float32)

scores = fused_rope_qk(Q, K, sin, cos)
print(scores.shape)   # (1, 4, 4)
```

## What the gate checks

The grader computes a reference implementation that first rotates `Q` and `K` explicitly and then performs the dot product.  
Your solution must produce scores whose maximum absolute element‑wise difference from this reference is at most $10^{-5}$:

$$
\max_{i,j} |\, \text{your}(i,j) - \text{reference}(i,j)\,| \le 10^{-5}.
$$

The gate metric used is `max_abs_err`.  A correct implementation will pass; a naïve or incorrect one will fail.
