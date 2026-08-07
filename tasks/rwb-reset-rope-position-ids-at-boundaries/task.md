## Context

Large language models often process multiple independent sequences in one packed batch to reduce padding overhead. The packed representation concatenates tokens from different sequences, so a token index in the buffer is not the same as its position inside its original sequence.

Rotary positional embeddings (RoPE) encode positions by rotating pairs of hidden dimensions. For a pair $(x_0, x_1)$ and angle $\theta_p$, the rotation is

$$
\begin{bmatrix}
x'_0 \\
x'_1
\end{bmatrix}
=
\begin{bmatrix}
\cos(\theta_p) & -\sin(\theta_p) \\
\sin(\theta_p) & \cos(\theta_p)
\end{bmatrix}
\begin{bmatrix}
x_0 \\
x_1
\end{bmatrix}.
$$

For packed sequences, the correct position for each segment restarts from zero. If a packed buffer contains sequence lengths $l_1, l_2, \dots, l_m$, then the position ids are

$$
[0,1,\dots,l_1-1,\;0,1,\dots,l_2-1,\;\dots,\;0,1,\dots,l_m-1].
$$

Attention must also be computed independently inside each sequence. For query matrix $Q$, key matrix $K$, and value matrix $V$, scaled dot-product attention is

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V.
$$

## Task

Implement `packed_rope_attention(q, k, v, cu_seqlens)`.

The inputs are packed list:

- `q`, `k`, and `v` have shape $(n, d)$ and dtype $float64$.
- `cu_seqlens` is a one-dimensional integer array of length $m+1$ containing cumulative sequence boundaries.

The function must:

1. Build position ids that restart from zero at every interval `[cu_seqlens[i], cu_seqlens[i+1])`.
2. Apply RoPE to both `q` and `k` using the position inside the current sequence.
3. Compute attention separately for every non-empty sequence.
4. Return the concatenated output array with shape $(n, d)$.

Use the RoPE frequency rule

$$
\theta_p = \frac{p}{10000^{2j/d}}
$$

for dimension pair index $j$.

## Example

```python

q = [[1.0] * 4 for _ in range(3)]
k = [[1.0] * 4 for _ in range(3)]
v = list(range(12)).reshape(3, 4)
cu_seqlens = [0, 2, 3]

out = packed_rope_attention(q, k, v, cu_seqlens)
# The first two rows attend together and the last row attends by itself.
```

## What the gate checks

The grader computes a Python oracle by unpacking every sequence, assigning positions from $0$ to $l_i-1$, applying RoPE, and running attention independently for each non-empty segment.

The returned tensor is compared to the oracle using

$$
\max_{i,j}|y_{i,j}-\hat{y}_{i,j}|.
$$

The value must be less than $10^{-5}$. A solution that uses global packed-buffer positions instead of resetting positions at boundaries will fail.
