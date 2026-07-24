## Context

PyTorch's `torch.nn.functional.scaled_dot_product_attention` accepts an
`enable_gqa=True` flag: when `K`/`V` carry fewer heads than `Q`, SDPA
broadcasts each KV head across the block of query heads that shares it,
instead of requiring the caller to pre-expand `K`/`V` into full-size
tensors themselves. With $n_q$ query heads and $n_{kv}$ key/value heads,
where $n_q$ is a multiple of $n_{kv}$, define the repeat factor

$$
r = \frac{n_q}{n_{kv}} .
$$

Query head $h$ (for $h = 0, \dots, n_q-1$) is served by KV head

$$
\mathrm{kv\_group}(h) = \left\lfloor \frac{h}{r} \right\rfloor,
$$

so query heads $0, \dots, r-1$ all use KV head $0$, query heads
$r, \dots, 2r-1$ all use KV head $1$, and so on — a **blocked** grouping,
not a cyclic/interleaved one (query head $1$ does *not* jump to KV head $1$
unless $r = 1$). For each query head $h$, attention is then ordinary scaled
dot-product attention against its assigned KV head:

$$
\mathrm{score}_h = \frac{Q_h\, K_{\mathrm{kv\_group}(h)}^\top}{\sqrt{d}},
\qquad
\mathrm{out}_h = \mathrm{softmax}(\mathrm{score}_h)\, V_{\mathrm{kv\_group}(h)} .
$$

## Task

Implement `enable_gqa_broadcast_attention(Q, K, V)`:

```python
def enable_gqa_broadcast_attention(Q, K, V):
    ...
```

Inputs use PyTorch SDPA's native axis order, `(batch, heads, seq, dim)`:
- `Q` has shape $(batch, n_q, seq_q, d)$.
- `K`, `V` have shape $(batch, n_{kv}, seq_k, d)$, with $n_q$ a positive
  integer multiple of $n_{kv}$.

Broadcast `K` and `V` up to $n_q$ heads following the **blocked** grouping
above (query head $h$ reads KV head $\lfloor h/r \rfloor$), then run scaled
dot-product attention per query head with scale $1/\sqrt{d}$. Return a
`float64` NumPy array of shape $(batch, n_q, seq_q, d)$.

Do not use a cyclic/`tile`-style assignment (query head $h$ reading KV head
$h \bmod n_{kv}$) — that groups the *wrong* query heads together whenever
$r > 1$.

## Example

```python
import numpy as np

rng = np.random.default_rng(0)
Q = rng.standard_normal((1, 6, 4, 8))   # 6 query heads
K = rng.standard_normal((1, 2, 4, 8))   # 2 KV heads -> r = 3
V = rng.standard_normal((1, 2, 4, 8))

out = enable_gqa_broadcast_attention(Q, K, V)
# out.shape == (1, 6, 4, 8)
# query heads 0,1,2 all attend against K[:,0]/V[:,0]
# query heads 3,4,5 all attend against K[:,1]/V[:,1]
```

## What the gate checks

The gate combines two kinds of cases, all built with `np.random.default_rng(0)`:

1. **Numeric cases** across several `(batch, n_q, n_kv, seq_q, seq_k, d)`
   configurations (including the degenerate $n_{kv}=n_q$ case and several
   $r>1$ groupings), compared against an oracle that repeats each KV head
   $r$ times with blocked semantics (`np.repeat(..., r, axis=1)`) and runs
   dense attention in `float64`.
2. **A grouping probe**: a case with a single key/value token per KV head
   and one distinct one-hot `V` row per KV head. Since softmax over a
   single key is always exactly $1$, the output for query head $h$ is
   *exactly* the one-hot vector belonging to `kv_group(h)` — this pins down,
   per head, exactly which KV group your implementation actually used. A
   cyclic (`h % n_kv`) grouping reproduces the wrong one-hot rows for every
   $r>1$ case and fails this probe outright.

Both kinds of cases are scored with `max_abs_err`; the maximum over every
case must stay below $10^{-5}$.
