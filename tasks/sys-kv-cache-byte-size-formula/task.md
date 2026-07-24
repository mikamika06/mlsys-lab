## Context

During autoregressive decoding a transformer keeps a **KV cache**: the
key and value activations for every past token, for every layer and every
KV head, so they don't need to be recomputed at each new step. For a model
with $L$ layers, $n_{kv}$ KV heads per layer, head dimension $d$, a cached
sequence length of $\text{seq}$ tokens, and a storage dtype of
$\text{dtype\_bytes}$ bytes per element, the total KV-cache size in bytes is
$$
\text{kv\_bytes}(n_{kv}) \;=\; 2 \cdot L \cdot n_{kv} \cdot d \cdot \text{seq} \cdot \text{dtype\_bytes},
$$
where the factor $2$ accounts for storing **both** K and V.

This formula is exactly why grouped-query attention (GQA, $n_{kv} < n_q$)
and multi-query attention (MQA, $n_{kv}=1$) shrink the KV cache: the byte
count scales with $n_{kv}$, not with the number of query heads $n_q$. Full
multi-head attention (MHA) is the special case $n_{kv} = n_q$. Comparing a
GQA/MQA configuration against the MHA baseline **at the same $n_q$** gives
$$
\text{ratio\_vs\_mha} \;=\; \frac{\text{kv\_bytes}(n_{kv})}{\text{kv\_bytes}(n_q)} \;=\; \frac{n_{kv}}{n_q} \;=\; \frac{1}{g}, \qquad g = \frac{n_q}{n_{kv}} \;(\text{the group factor}).
$$
E.g. 8 query heads grouped down to 2 KV heads ($g=4$) cuts the KV cache to
exactly $1/4$ of the MHA baseline's size — independent of $L$, $d$,
$\text{seq}$, or $\text{dtype\_bytes}$, since those factors are identical
on both sides of the ratio and cancel out.

## Task

Implement:

```python
def kv_cache_size(L: int, n_q: int, n_kv: int, d: int, seq: int,
                   dtype_bytes: int) -> tuple[int, float]:
    ...
```

* `L` — number of transformer layers.
* `n_q` — number of query heads per layer.
* `n_kv` — number of KV heads per layer (`n_q` is a multiple of `n_kv`).
* `d` — head dimension.
* `seq` — cached sequence length (number of tokens).
* `dtype_bytes` — bytes per stored element (e.g. `2` for fp16/bf16).

Return a tuple `(kv_bytes, ratio_vs_mha)`:

* `kv_bytes` — the total KV-cache size in bytes for this config:
  $2 \cdot L \cdot n_{kv} \cdot d \cdot \text{seq} \cdot \text{dtype\_bytes}$.
* `ratio_vs_mha` — the ratio of `kv_bytes` at this `n_kv` to `kv_bytes` at
  the full-MHA baseline `n_kv = n_q` (same `L`, `d`, `seq`,
  `dtype_bytes`): should equal $n_{kv}/n_q$.

## Example

```python
kv_cache_size(L=32, n_q=8, n_kv=2, d=128, seq=4096, dtype_bytes=2)
# kv_bytes = 2 * 32 * 2 * 128 * 4096 * 2 = 134,217,728
# ratio_vs_mha = n_kv / n_q = 2 / 8 = 0.25
# -> (134217728, 0.25)
```

## What the gate checks

Two gates run over several seeded random configs (spanning MHA, MQA, and
general GQA group factors):

* **size_ratio** — the worst-case absolute difference between your
  returned `ratio_vs_mha` and the true $n_{kv}/n_q$ across all trials, must
  be $\le 10^{-12}$.
* **rel_err** — the worst-case relative error between your returned
  `kv_bytes` and the exact formula value across all trials, must be
  $\le 10^{-12}$. This catches implementations that get the *ratio* right
  by accident (e.g. forgetting the factor of `2` for K+V, which cancels out
  of the ratio) but return the wrong absolute byte count.

Both gates must pass; any exception fails both.
