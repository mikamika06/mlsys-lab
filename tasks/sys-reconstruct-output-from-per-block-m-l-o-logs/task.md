## Context

Online (streaming) softmax processes keys/values one block at a time,
keeping only three running statistics instead of the full score matrix:
a running max $m$, a running softmax denominator $l$, and a running
weighted value accumulator $o$ (all *unnormalized* — the true output is
only recovered at the very end as $o / l$).

The same idea works the other way around: if $K$ workers each compute
their own **local** triple $(m_k, l_k, o_k)$ over their own block of keys
and values, with no coordination —

$$
m_k = \max_j s_{kj}, \qquad
l_k = \sum_j e^{\,s_{kj} - m_k}, \qquad
o_k = \sum_j e^{\,s_{kj} - m_k}\, v_{kj}
$$

— those $K$ local triples can be losslessly **merged** into the single
correct *global* softmax output, without ever seeing the raw scores or
values again. For a running $(m, l, o)$ and a new block's $(m_k, l_k, o_k)$:

$$
m' = \max(m, m_k), \qquad
l' = l\, e^{\,m - m'} + l_k\, e^{\,m_k - m'}, \qquad
o' = o\, e^{\,m - m'} + o_k\, e^{\,m_k - m'}
$$

This merge is associative and commutative — any order of combination
gives the same exact result — because it is just a numerically-stable
rewrite of $\sum_k e^{m_k - m'} \big(\cdot\big)$ factored out of a single
global log-sum-exp. After merging every block, the reconstructed output
is $o_{\text{final}} / l_{\text{final}}$.

## Task

Implement `reconstruct_attention_from_block_logs`:

```python
def reconstruct_attention_from_block_logs(block_m: list[float], block_l: list[float], block_o: list[list[float]]) -> list[float]:
    ...
```

- `block_m`: shape `(K,)` float64 — each block's local max score.
- `block_l`: shape `(K,)` float64 — each block's local softmax
  denominator (computed using only that block's own scores).
- `block_o`: shape `(K, d)` float64 — each block's local weighted value
  accumulator (computed using only that block's own scores/values).

**No raw scores or values are available** — these three per-block logs
are all that survived (exactly what an online-softmax / flash-attention
worker would keep after discarding its raw block). Merge all $K$ blocks
using the rule above (in any order — it's associative) and return the
final normalized output vector, shape `(d,)`.

## Example

```python

# 2 blocks of one key each, values are 1-D (d=1), for hand-checkable numbers
block_m = [1.0, 2.0]                 # local maxes
block_l = [1.0, 1.0]                 # each block: single key -> l = exp(0) = 1
block_o = [[1.0], [2.0]]              # o_k = exp(0) * v_k, with v = [1.0, 2.0]

out = reconstruct_attention_from_block_logs(block_m, block_l, block_o)
# global scores are [1.0, 2.0] -> softmax = [exp(-1), 1] / (exp(-1)+1)
#                                 ≈ [0.2689, 0.7311]
# out ≈ 0.2689*1.0 + 0.7311*2.0 ≈ [1.7311]
```

## What the gate checks

The grader builds several cases (varying block count and block sizes,
including singleton blocks and a single-block case) from real per-token
scores and values, computes each block's local `(m, l, o)` — the only
things passed to your function — and independently computes the true
global attention output directly from the *raw, concatenated* scores and
values (a plain dense softmax, never touching the merge algebra).

`max_abs_err` is the worst-case max elementwise absolute difference
between your reconstructed output and this independently-computed dense
reference, across all cases (must be `< 1e-6`). Merging in the wrong
order matters for correctness only if done incorrectly (e.g. rescaling
`o`/`l` by the wrong factor, or forgetting to rescale the running
accumulator when a later block's max exceeds it) — those bugs show up
directly as a large error here.
