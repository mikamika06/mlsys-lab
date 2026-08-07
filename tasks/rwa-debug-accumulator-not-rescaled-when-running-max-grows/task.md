## Context

FlashAttention-style kernels never materialize the full $n\times n$
attention matrix. For a single query $q\in\mathbb{R}^d$ against keys/values
$K, V \in \mathbb{R}^{n\times d}$, they stream over $K/V$ in blocks and
maintain three running statistics:

- $m$ — the running max of $q\cdot k_i/\sqrt d$ seen so far,
- $l$ — the running softmax normalizer (sum of $\exp(\text{score}-m)$),
- $O$ — an **unnormalized** output accumulator.

Processing block $j$ with local scores $s_j = q K_j^\top/\sqrt d$:

$$
m_{\text{new}} = \max(m_{\text{old}}, \max(s_j)), \qquad
\alpha = \exp(m_{\text{old}} - m_{\text{new}})
$$

$$
l_{\text{new}} = \alpha\, l_{\text{old}} + \sum_i \exp\big(s_{j,i} - m_{\text{new}}\big)
$$

$$
O_{\text{new}} = \alpha\, O_{\text{old}} + \sum_i \exp\big(s_{j,i} - m_{\text{new}}\big)\, V_{j,i}
$$

The rescale factor $\alpha$ is what lets the algorithm process blocks
one at a time while still producing output identical (up to floating
point) to the dense softmax $\operatorname{softmax}(qK^\top/\sqrt d)\,V$
computed with the *final* global max. Crucially, $\alpha$ must be applied
to **both** $l$ **and** $O$ — every previously-accumulated block was
weighted against the max known *at the time it was processed*, and that
weighting only stays correct if it gets corrected every time the max
grows.

## Task

`tasks/rwa-debug-accumulator-not-rescaled-when-running-max-grows/starter.py`
contains a broken `tiled_online_softmax_attention`. It correctly rescales
`l` by `alpha` on every block, but **forgets to rescale `O`** — so `O`
silently accumulates each block's contribution weighted against that
block's own stale local max instead of the sequence's final global max.

Fix the implementation so it matches the dense reference:

```python
def tiled_online_softmax_attention(q: list[float], K: list[list[float]], V: list[list[float]], block_size: int) -> list[float]:
    ...
```

- `q`: `(d,)`. `K`, `V`: `(n, d)`. `block_size`: positive int (`n` need
  not be a multiple of it — the last block may be shorter).
- Process blocks left to right, in order, updating `m`, `l`, `O` with the
  recurrence above (`m` starts at `-inf`, `l` at `0`, `O` at zeros — note
  `exp(-inf - m_new)` naturally evaluates to `0.0` for the first block, no
  special-casing needed).
- Return `O / l`, shape `(d,)`.

## Example

```python

import random; rng = random.Random(0)
q = rng.standard_normal(4)
K = rng.standard_normal((20, 4))
V = rng.standard_normal((20, 4))

out = tiled_online_softmax_attention(q, K, V, block_size=6)

# reference: dense softmax over the whole sequence at once
scores = [sum(q_i * k_i for q_i, k_i in zip(q, row)) / math.sqrt(4) for row in K]
scores -= scores.max()
p = [math.exp(s) for s in scores]; p = [x / sum(p) for x in p]
ref = p @ V

assert max(abs(o - r) for o, r in zip(out, ref)) < 1e-4
```

## What the gate checks

The grader runs one deterministic case with a strong outlier key placed
in a later block (guaranteeing a large accumulator rescale is needed
after the first block already contributed real mass to `O`), plus several
seeded random multi-block cases, and compares your output to a dense
`softmax(qK^T/√d) @ V` computed directly in Python — never calling your
function, never hardcoding an expected value.

`max_abs_err` is the worst per-case max-abs-error across all cases and
must be `<= 1e-4`. The buggy starter leaves earlier blocks weighted
against their own stale local max, which produces an output error far
above this threshold on any case with more than one block and a
mid-sequence max increase.
