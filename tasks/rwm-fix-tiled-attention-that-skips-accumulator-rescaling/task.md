## Context

FlashAttention-style kernels never materialize the full $n_q \times n_k$
attention matrix. Instead they stream $K$/$V$ in blocks and keep, for every
query row, a running maximum score $m$, a running softmax denominator $l$,
and a running numerator accumulator $\mathrm{acc}$. For a block with scores
$s = Q_{\text{blk}} K_{\text{blk}}^\top$ and block max $m_{\text{blk}} =
\max_j s_j$, the running max is updated as

$$
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{blk}}).
$$

The previous accumulators $l_{\text{old}}$ and $\mathrm{acc}_{\text{old}}$
were computed relative to $m_{\text{old}}$. Whenever $m_{\text{new}} \ne
m_{\text{old}}$ they are now on the wrong scale and **must be rescaled**
by $\alpha = \exp(m_{\text{old}} - m_{\text{new}})$ before the new block's
contribution is folded in:

$$
l_{\text{new}} = \alpha\, l_{\text{old}} + \sum_j e^{s_j - m_{\text{new}}},
\qquad
\mathrm{acc}_{\text{new}} = \alpha\, \mathrm{acc}_{\text{old}} + \sum_j e^{s_j - m_{\text{new}}} v_j .
$$

The final output for a query row is $\mathrm{acc} / l$, which equals exact
softmax attention regardless of block size — **provided every max update is
paired with a rescale.**

## Task

The supplied `tiled_attention_forward` has a bug: it updates the running max
`m` for every block, but never rescales the `l` and `acc` accumulators onto
the new max before adding the new block's terms. Fix it.

```python
def tiled_attention_forward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], block_size: int) -> list[list[float]]:
    ...
```

* `Q` — queries, shape $(n_q, d)$.
* `K` — keys, shape $(n_k, d)$.
* `V` — values, shape $(n_k, d_v)$.
* `block_size` — number of key/value rows processed per streaming block
  (the last block may be shorter; `block_size` need not divide `n_k`).

Process $K$/$V$ in consecutive blocks of `block_size` rows, maintaining the
running $(m, l, \mathrm{acc})$ recurrence above **with the rescale applied on
every max update**. Return the output $O \in \mathbb{R}^{n_q \times d_v}$
where row $i$ is $\mathrm{acc}_i / l_i$ — this must equal plain non-causal
softmax attention $O = \mathrm{softmax}(QK^\top)V$ to numerical precision,
independent of `block_size`.

## Example

```python

Q = [[1.0, 0.0]]
K = [[0.0, 1.0], [1.0, 0.0], [5.0, 0.0]]   # last block has the largest score
V = [[1.0], [2.0], [10.0]]

out = tiled_attention_forward(Q, K, V, block_size=1)
# The last block (score 5.0) dominates the softmax, so out should be very
# close to [[10.0]] regardless of how the earlier blocks were processed.
# The buggy version under-weights that last block because its accumulators
# were never rescaled onto the new, larger max.
```

## What the gate checks

The grader computes the reference output directly from the definition,

$$
O = \mathrm{softmax}(QK^\top)\,V,
$$

with a numerically-stable row-wise softmax, for several `(Q, K, V,
block_size)` cases — including cases where key rows are scaled so that later
blocks contain larger scores than earlier ones, several block sizes
(including ones that don't evenly divide `n_k`), and a single-query case.
Your output must match the oracle with a maximum absolute error of at most
$10^{-6}$ on every case. The buggy starter (missing rescale) fails this gate
because it silently under-weights any block that arrives after the running
max has already increased.
