## Context

Grouped-query attention (GQA) and multi-query attention (MQA) shrink the
KV cache by giving the model fewer key/value heads ($H_{kv}$) than query
heads ($H_q$). $H_q$ must be a multiple of $H_{kv}$; each KV head is
shared by a contiguous **group** of
$n_{\text{rep}} = H_q / H_{kv}$ query heads. MQA is the extreme case
$H_{kv} = 1$ (every query head shares the single KV head); ordinary MHA
is $H_{kv} = H_q$ ($n_{\text{rep}} = 1$, no sharing).

There is no separate "grouped attention math" — a KV head is simply
**broadcast** (repeated) across its group before running ordinary
per-head attention. Query head $h$ (0-indexed) reads from KV head
$\lfloor h / n_{\text{rep}} \rfloor$:

$$
\text{out}_h = \mathrm{softmax}\!\left(\frac{Q_h K_{\lfloor h/n_{\text{rep}}\rfloor}^\top}{\sqrt{d}}\right) V_{\lfloor h/n_{\text{rep}}\rfloor}, \qquad h = 0, \dots, H_q - 1
$$

## Task

Implement `gqa_broadcast_attention`:

```python
def gqa_broadcast_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    ...
```

- `q`: `(H_q, n, d)` float64 — per-head queries.
- `k`, `v`: `(H_kv, n, d)` float64 — per-head keys/values, with
  `H_q % H_kv == 0`.

Return the `(H_q, n, d)` attention output: for each query head $h$,
broadcast KV head $\lfloor h / n_{\text{rep}} \rfloor$
(where $n_{\text{rep}} = H_q / H_{kv}$) and run standard (non-causal)
scaled dot-product attention $\mathrm{softmax}(QK^\top/\sqrt{d})V$ using
that query head against the broadcast key/value head.

## Example

```python
import numpy as np

H_q, H_kv, n, d = 4, 2, 5, 8
rng = np.random.default_rng(0)
q = rng.normal(size=(H_q, n, d))
k = rng.normal(size=(H_kv, n, d))
v = rng.normal(size=(H_kv, n, d))

out = gqa_broadcast_attention(q, k, v)
# n_rep = 4 // 2 = 2: query heads 0,1 both read KV head 0;
# query heads 2,3 both read KV head 1.
# out.shape == (4, 5, 8)
```

## What the gate checks

The grader builds several `(q, k, v)` scenarios from a seeded NumPy
generator — plain MHA (`H_kv == H_q`), GQA with a few different group
sizes, and MQA (`H_kv = 1`) — and computes the reference output
independently in float64: explicitly repeat each KV head
`n_rep = H_q // H_kv` times (`numpy.repeat` along the head axis, so KV
head $i$ maps to query heads $[i \cdot n_{\text{rep}}, (i+1)\cdot n_{\text{rep}})$),
then run ordinary per-head scaled dot-product attention — never calling
your function.

`max_abs_err` is the worst-case elementwise absolute error between your
output and the oracle's, across every scenario, and the gate requires
`<= 1e-5`. Broadcasting KV heads in the wrong order (e.g. tiling instead
of repeat-interleave, so query head 0 pairs with KV head 0 but query
head 1 pairs with KV head 1 in an `H_kv=2, H_q=4` case instead of both
reading KV head 0), reusing the same KV head for every query head
regardless of group, or forgetting the $1/\sqrt{d}$ scale will all
produce a mismatch for any scenario where `H_kv < H_q`.
