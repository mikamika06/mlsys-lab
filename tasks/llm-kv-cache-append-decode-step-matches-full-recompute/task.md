## Context

Autoregressive transformers generate tokens one at a time. At decode step $t$, self-attention for the new token needs to attend to keys and values from all previous positions.

A transformer attention head computes

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V .
$$

During decoding, recomputing $K$ and $V$ for every previous token at every step is wasteful. A KV-cache stores previous key and value vectors. When a new token arrives, its key and value are appended to the cache:

$$
K_{\mathrm{cache}} \leftarrow
\begin{bmatrix}
K_{\mathrm{cache}}\\
k_t
\end{bmatrix},
\qquad
V_{\mathrm{cache}} \leftarrow
\begin{bmatrix}
V_{\mathrm{cache}}\\
v_t
\end{bmatrix}.
$$

The output for the current token should be identical to recomputing attention over all tokens seen so far.

## Task

Implement `decode_steps`:

```python
def decode_steps(
    x: list[list[float]],
    Wq: list[list[float]],
    Wk: list[list[float]],
    Wv: list[list[float]],
) -> list[list[float]]:
    ...
```

Inputs:

- `x`: a list of shape $(n, d)$ containing token representations.
- `Wq`, `Wk`, `Wv`: list of shape $(d, d)$ containing projection matrices.

Return a list of shape $(n, d)$ containing the attention output for each decode step.

For each position $t$, compute the query from the current token, append the current key and value to the KV-cache, and attend over the complete cache. The returned row $t$ must match a full recomputation over `x[:t+1]`.

Use Python operations. The implementation should maintain the cache rather than rebuilding the whole history for every step.

## Example

```python

x = [[1.0, 0.0], [0.0, 1.0]]
Wq = [[1.0 if i == j else 0.0 for j in range(2)] for i in range(2)]
Wk = [[1.0 if i == j else 0.0 for j in range(2)] for i in range(2)]
Wv = [[1.0 if i == j else 0.0 for j in range(2)] for i in range(2)]

y = decode_steps(x, Wq, Wk, Wv)
```

The first row attends only to the first token. The second row attends to both tokens because the cache contains both key/value pairs.

## What the gate checks

The gate builds a Python reference implementation that performs full recomputation for every decode step. It compares the submitted implementation against this oracle.

The maximum absolute element error

$$
\max_i |y_i-\hat{y}_i|
$$

must be at most $10^{-5}$. An implementation that ignores previous cached tokens or updates the cache incorrectly will fail this check.
