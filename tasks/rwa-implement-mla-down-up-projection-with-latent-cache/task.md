## Context

Standard multi-head attention caches a full-size $K$ and $V$ per token —
`num_heads * d_head` numbers each. **Multi-head Latent Attention (MLA)**
shrinks that cache by routing keys and values through a shared low-rank
bottleneck: instead of caching $K$ and $V$ directly, a token's hidden
state $x$ is **down-projected** once into a small shared latent

$$
c_{KV} = x \, W_{\text{down}} \in \mathbb{R}^{r}, \qquad r = \text{kv\_lora\_rank}
$$

and *only $c_{KV}$ is cached* — $r$ numbers per token instead of
$2 \cdot \text{num\_heads} \cdot d_{\text{head}}$. Whenever $K$ or $V$ is
actually needed for an attention computation, they are **up-projected**
back out of the cached latent, per head:

$$
K = c_{KV} \, W_{\text{up}}^{K}, \qquad V = c_{KV} \, W_{\text{up}}^{V},
\qquad W_{\text{up}}^{K}, W_{\text{up}}^{V} \in \mathbb{R}^{r \times (\text{num\_heads} \cdot d_{\text{head}})}
$$

Because $W_{\text{down}}$, $W_{\text{up}}^{K}$, and $W_{\text{up}}^{V}$
are exactly the matrices a standard $K$/$V$ projection would have been
factored into, the down-then-up path reproduces standard multi-head
attention's output bit-for-bit (up to floating point) — the only thing
that changes is *how much gets cached between the projection and the
attention computation*.

## Task

Implement `mla_forward`:

```python
def mla_forward(
    x: np.ndarray,
    W_Q: np.ndarray,
    W_down_kv: np.ndarray,
    W_up_K: np.ndarray,
    W_up_V: np.ndarray,
    num_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    ...
```

- `x`: `(n, d_model)` float64 hidden states.
- `W_Q`: `(d_model, num_heads * d_head)` query projection.
- `W_down_kv`: `(d_model, kv_lora_rank)` shared KV down-projection.
- `W_up_K`, `W_up_V`: `(kv_lora_rank, num_heads * d_head)` per-head up-projections.
- `num_heads`: number of attention heads (`d_head` is inferred from
  `W_Q.shape[1] // num_heads`).

Return `(out, c_kv)`:

- `c_kv` — the cached latent, `x @ W_down_kv`, shape `(n, kv_lora_rank)`.
  This is the **only** array that would need to be kept around across
  decode steps in a real KV cache.
- `out` — the `(n, num_heads * d_head)` multi-head self-attention output:
  project `Q = x @ W_Q`; up-project `K = c_kv @ W_up_K`,
  `V = c_kv @ W_up_V`; reshape all three into `(num_heads, n, d_head)`;
  run standard (non-causal) scaled dot-product attention
  $\mathrm{softmax}(QK^\top/\sqrt{d_{\text{head}}})V$ **independently per
  head**; concatenate the heads back into `(n, num_heads * d_head)` in
  head order.

## Example

```python
import numpy as np

n, d_model, num_heads, d_head, r = 5, 8, 2, 4, 6
rng = np.random.default_rng(0)
x = rng.normal(size=(n, d_model))
W_Q = rng.normal(size=(d_model, num_heads * d_head))
W_down_kv = rng.normal(size=(d_model, r))
W_up_K = rng.normal(size=(r, num_heads * d_head))
W_up_V = rng.normal(size=(r, num_heads * d_head))

out, c_kv = mla_forward(x, W_Q, W_down_kv, W_up_K, W_up_V, num_heads)
# c_kv.shape == (5, 6)              -- only the latent, not full K/V
# out.shape  == (5, 8)              -- num_heads * d_head
```

## What the gate checks

The grader builds several `(x, W_Q, W_down_kv, W_up_K, W_up_V, num_heads)`
scenarios from a seeded NumPy generator (varying `d_model`, `num_heads`,
`d_head`, and `kv_lora_rank`) and computes two references independently
in float64: `c_kv_ref = x @ W_down_kv`, and `out_ref` via **standard
multi-head attention** built directly from the same matrices —
`K_full = c_kv_ref @ W_up_K`, `V_full = c_kv_ref @ W_up_V`, per-head
`softmax(QK^T/sqrt(d_head))V` — never calling your function.

Two gates apply: `max_abs_err` is the worst-case elementwise absolute
error between your `out` and `out_ref` (must be `<= 1e-4`), and
`latent_ok` is `1.0` only if your `c_kv` has exactly shape
`(n, kv_lora_rank)` *and* matches `x @ W_down_kv` to near machine
precision (else `0.0`). Recomputing full per-head `K`/`V` straight from
`x` (skipping the down-projection, so the cache would secretly still hold
full-size keys/values) will pass `max_abs_err` but fail `latent_ok`;
up-projecting from the wrong matrix, mixing up heads, or forgetting the
$1/\sqrt{d_{\text{head}}}$ scale will fail `max_abs_err`.
