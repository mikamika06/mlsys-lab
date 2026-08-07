## Context

Multi-head attention (MHA) gives every query head its own key/value head:
$n_q$ query heads, $n_q$ KV heads. That means the KV cache (which must be
kept resident during autoregressive decoding) grows linearly with $n_q$,
which becomes the memory bottleneck at long context lengths.

**Grouped-query attention (GQA)** shrinks the KV cache by using fewer KV
heads than query heads: $n_{kv} < n_q$, with $n_q$ evenly divisible by
$n_{kv}$. Query heads are split into $n_{kv}$ contiguous, equally-sized
groups of size
$$
g = \frac{n_q}{n_{kv}},
$$
and every query head in group $k$ (query heads $k\cdot g, \dots,
(k+1)\cdot g - 1$) attends using the **same** KV head $k$:
$$
O_h = \operatorname{softmax}\!\Big(\frac{Q_h K_{\lfloor h/g \rfloor}^\top}{\sqrt{d}}\Big)\, V_{\lfloor h/g \rfloor},
\qquad h = 0, \dots, n_q-1 .
$$

Two special cases fall out of this one formula: $n_{kv} = n_q$ (so $g=1$,
every query head gets its own KV head) reduces exactly to ordinary MHA;
$n_{kv} = 1$ (so $g = n_q$, every query head shares the single KV head) is
**multi-query attention (MQA)**.

A common implementation bug is broadcasting the KV heads with the wrong
grouping: e.g. assigning query head $h$ to KV head $h \bmod n_{kv}$
("interleaved"/tile-style) instead of $\lfloor h/g \rfloor$
("blocked"/repeat-style). Both happen to agree when $n_{kv}=n_q$ (every
group has size 1, so $h \bmod n_{kv} = \lfloor h/g\rfloor = h$), which is
exactly why that bug can hide behind a passing MHA-only sanity check.

## Task

Implement:

```python
def gqa_attention(Q: list[list[list[float]]], K: list[list[list[float]]], V: list[list[list[float]]]) -> list[list[list[float]]]:
    ...
```

* `Q` — float array of shape $(n_q, n, d)$: $n_q$ query heads, sequence
  length $n$, head dim $d$.
* `K`, `V` — float arrays of shape $(n_{kv}, n, d)$, with $n_q$ divisible by
  $n_{kv}$.

For every query head $h$, run scaled-dot-product attention (scale
$1/\sqrt{d}$, numerically-stable softmax) against KV head
$\lfloor h / g \rfloor$ where $g = n_q / n_{kv}$, i.e. query heads are
grouped into $n_{kv}$ contiguous blocks of size $g$, each block sharing one
KV head. Return `O` of shape $(n_q, n, d)$.

## Example

With $n_q=4$, $n_{kv}=2$ (so $g=2$): query heads $0,1$ both attend to KV
head $0$; query heads $2,3$ both attend to KV head $1$.

```python
O = gqa_attention(Q, K, V)   # Q: (4,n,d), K,V: (2,n,d)
# O[0], O[1] were computed against K[0], V[0]
# O[2], O[3] were computed against K[1], V[1]
```

## What the gate checks

A single gate, **max_abs_err**, generates several seeded random instances —
including plain MHA ($n_{kv}=n_q$), MQA ($n_{kv}=1$), and general GQA with
$n_{kv}$ strictly between $1$ and $n_q$ — computes the reference output
per-head directly with Python (looping over query heads, indexing KV head
$\lfloor h/g\rfloor$, doing a numerically-stable softmax), and compares it
element-wise to your function's output. The maximum absolute error across
all trials must be $\le 10^{-5}$; any exception or wrong output shape
counts as a failing (`1e9`) error.
