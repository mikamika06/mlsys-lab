## Context

StreamingLLM-style attention keeps a subset of tokens in a KV cache. Rotary position embeddings (RoPE) use a position id to rotate query and key vectors before attention. In this setting, the position id is the token's cache slot, not its original position in the full sequence.

For a vector pair $(x_{2i}, x_{2i+1})$, RoPE rotates the pair by angle

$$
\theta_i(p) = p \cdot 10000^{-2i/d},
$$

where $p$ is the position id and $d$ is the embedding dimension. The rotated pair is

$$
\begin{bmatrix}
x'_{2i}\\
x'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
\cos(\theta_i(p)) & -\sin(\theta_i(p))\\
\sin(\theta_i(p)) & \cos(\theta_i(p))
\end{bmatrix}
\begin{bmatrix}
x_{2i}\\
x_{2i+1}
\end{bmatrix}.
$$

If the kept original token indices are $s_0, s_1, \dots, s_{k-1}$, StreamingLLM remaps them to cache slots

$$
p_j = j,\quad j \in \{0,\dots,k-1\}.
$$

Attention is then computed using the rotated queries and keys:

$$
\mathrm{softmax}\left(\frac{Q_r K_r^\top}{\sqrt{d}}\right)V.
$$

## Task

Implement `streaming_rope_attention(q, k, v, kept_indices, theta=10000.0)`.

Arguments:

- `q`, `k`, and `v` are list of shape $(n,d)$ containing the original sequence tensors.
- `kept_indices` is a 1-D integer list containing the original token indices kept in the cache.
- `theta` is the RoPE base.

Return the attention output for the kept tokens. The function must select the kept rows from `q`, `k`, and `v`, remap their RoPE positions to cache slots $0,1,\dots,k-1$, apply RoPE to queries and keys using those positions, and compute attention in float64.

Use Python operations. The embedding dimension $d$ is even.

## Example

```python

q = [[1., 0.], [0., 1.], [1., 1.]]
k = [[1., 0.], [1., 1.], [0., 1.]]
v = [[1., 2.], [3., 4.], [5., 6.]]

out = streaming_rope_attention(
    q, k, v, [0, 2]
)
```

The two kept tokens have cache positions $0$ and $1$, even though their original positions were $0$ and $2$.

## What the gate checks

The gate builds a Python oracle that selects kept tokens, assigns cache-slot position ids, applies RoPE, and computes float64 attention. The returned array is compared with the oracle using maximum absolute error:

$$
\max_i |y_i-\hat{y}_i| < 10^{-5}.
$$

Using original sequence indices as RoPE positions produces a numerically different attention result and fails the gate.
