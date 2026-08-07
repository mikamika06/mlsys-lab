## Context

Grouped-query attention (GQA), as used in Llama 2/3, Mistral, and similar
production models, stores **fewer** key/value heads than query heads in the
KV cache. If there are $n_q$ query heads and $n_{kv}$ key/value heads with
$n_q$ a multiple of $n_{kv}$, each KV head is shared by

$$
r = \frac{n_q}{n_{kv}}
$$

adjacent query heads. At attention time, the small $K, V$ tensors are
expanded back up to $n_q$ heads by *repeating each KV head $r$ times
consecutively* — exactly `repeat_interleave` semantics, not tiling the whole
KV head sequence:

$$
K^{\text{exp}}_h = K_{\lfloor h / r \rfloor}, \qquad
V^{\text{exp}}_h = V_{\lfloor h / r \rfloor}, \qquad h = 0, \dots, n_q - 1 .
$$

So query heads $0, 1, \dots, r-1$ all read KV head $0$; query heads
$r, \dots, 2r-1$ all read KV head $1$; and so on. After expansion, standard
scaled dot-product attention runs per query head exactly as in ordinary
multi-head attention:

$$
\mathrm{score}_h = \frac{Q_h (K^{\text{exp}}_h)^\top}{\sqrt{d}}, \qquad
\mathrm{out}_h = \mathrm{softmax}(\mathrm{score}_h)\, V^{\text{exp}}_h .
$$

The degenerate case $n_{kv} = n_q$ (so $r = 1$) makes $K^{\text{exp}} = K$
and $V^{\text{exp}} = V$ exactly, i.e. this collapses to plain MHA. The KV
cache only stores $n_{kv}$ heads instead of $n_q$, so its memory relative to
a full-MHA cache of the same shape is

$$
\text{memory\_ratio} = \frac{n_{kv}}{n_q} .
$$

## Task

Implement `gqa_head_expansion_attention(Q, K, V)`:

```python
def gqa_head_expansion_attention(Q: list[list[list[list[float]]]], K: list[list[list[list[float]]]], V: list[list[list[list[float]]]]) -> tuple[list[list[list[list[float]]]], float]:
    ...
```

Inputs are list:
- `Q` has shape $(batch, seq_q, n_q, d)$.
- `K`, `V` have shape $(batch, seq_k, n_{kv}, d)$, with $n_q$ a positive
  integer multiple of $n_{kv}$.

Steps:
1. Compute $r = n_q / n_{kv}$.
2. Expand `K` and `V` from $n_{kv}$ heads to $n_q$ heads by repeating each
   KV head $r$ times consecutively along the head axis (`repeat_interleave`,
e.g. `[x for x in ... for _ in range(r)]` — **not** `tile`, which would cycle
   through all KV heads before repeating any of them).
3. Run standard scaled dot-product attention between `Q` and the expanded
   `K`/`V`, with scale $1/\sqrt{d}$.

Return a tuple `(output, memory_ratio)`:
- `output`: `float64` list of shape $(batch, seq_q, n_q, d)$.
- `memory_ratio`: a Python float equal to $n_{kv} / n_q$.

Vectorize with Python; no explicit Python loop over heads/batch/sequence
positions is required.

## Example

```python

rng = random.Random(0)
Q = rng.standard_normal((1, 3, 4, 8))   # 4 query heads
K = rng.standard_normal((1, 3, 2, 8))   # 2 KV heads -> r = 2
V = rng.standard_normal((1, 3, 2, 8))

out, ratio = gqa_head_expansion_attention(Q, K, V)
# out.shape == (1, 3, 4, 8)
# ratio == 0.5          <- KV cache is half the size of full MHA

# degenerate case: n_kv == n_q reproduces plain MHA exactly
Kf = rng.standard_normal((1, 3, 4, 8))
Vf = rng.standard_normal((1, 3, 4, 8))
out_mha, ratio_mha = gqa_head_expansion_attention(Q, Kf, Vf)
# ratio_mha == 1.0
```

## What the gate checks

The gate builds several seeded `(Q, K, V)` cases with `random.Random(0)`
across different batch sizes, sequence lengths, and $(n_q, n_{kv})$ pairs,
including the degenerate $n_{kv} = n_q$ case (must reproduce plain MHA
exactly) and multiple $r > 1$ groupings. For each case it independently
expands `K`/`V` with `repeat_interleave` semantics and runs scaled
dot-product attention. Your `output` is compared against the oracle
element-for-element (`max_abs_err`, threshold $10^{-5}$); your
`memory_ratio` is checked for exact equality against $n_{kv}/n_q$
(`size_ratio == 1.0` gate). Expanding with `tile` instead of
`repeat_interleave` (which changes which query heads pair with which KV
head whenever $r > 1$) or getting the ratio upside down (e.g. $n_q/n_{kv}$)
fails the gate.
