## Context

FlashAttention never materializes the full softmax probability matrix $P$
in memory. During the forward pass it streams over blocks of keys/values
and, for each query row $i$, keeps only two running scalars instead of the
whole row of scores:

$$
m_i = \max_j S_{ij}, \qquad
l_i = \sum_j \exp\!\big(S_{ij} - m_i\big),
$$

where $S = QK^\top / \sqrt{d}$ is the scaled attention score matrix. $m_i$
is the row's max (used for numerically-stable exponentiation) and $l_i$ is
the row's softmax normalizer *after* that max-shift — together, $(m_i,
l_i)$ are exactly enough to reconstruct the full row of softmax
probabilities later, without ever storing $P$ and without a second pass to
find the max again:

$$
P_{ij} = \frac{\exp(S_{ij} - m_i)}{l_i}.
$$

This identity is what makes FlashAttention's backward pass (and gradient
checkpointing / activation recomputation in general) memory-efficient: $Q$,
$K$, $V$ and the tiny per-row $(m, l)$ pair are the only things kept from
the forward pass; $P$ and the output $O = PV$ are recomputed on demand from
them, cheaply and exactly.

## Task

Implement `flash_forward_reconstruct(Q, K, V, m, l)`:

```python
def flash_forward_reconstruct(Q, K, V, m, l):
    ...
```

Inputs are list:
- `Q` has shape $(n, d)$.
- `K` has shape $(k, d)$.
- `V` has shape $(k, d_v)$.
- `m` has shape $(n,)$ — the saved per-row max used when the scores were
  originally exponentiated.
- `l` has shape $(n,)$ — the saved per-row normalizer, consistent with that
  same `m` (i.e. $l_i = \sum_j \exp(S_{ij} - m_i)$ for the row's true
  scores $S_i$).

Recompute $S = QK^\top/\sqrt{d}$, then reconstruct $P$ and $O$ **using the
supplied `m` and `l` directly** — do not recompute your own row max or row
sum from $S$ and discard the given statistics; the whole point is that
`(m, l)` are already sufficient. Return a tuple `(P, O)`:

- `P`: `float64` array of shape $(n, k)$, with $P_{ij} = \exp(S_{ij} -
  m_i)/l_i$. Every row of `P` must sum to $1$.
- `O`: `float64` array of shape $(n, d_v)$, equal to `P @ V`.

## Example

```python

Q = [[1.0, 0.0], [0.0, 1.0]]
K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
V = [[1.0], [2.0], [3.0]]

S = [[sum(a * b for a, b in zip(row_q, row_k)) / math.sqrt(2) for row_k in K] for row_q in Q]
m = S.max(axis=1)
l = [sum(math.exp(s - mi) for s in row) for row, mi in zip(S, m)]

P, O = flash_forward_reconstruct(Q, K, V, m, l)
# P.sum(axis=1) is (approximately) [1.0, 1.0]
```

## What the gate checks

The gate builds several random `(Q, K, V)` cases with
`random.Random(0)`. For each case it computes two kinds of
`(m, l)` pairs to pass in:

1. The **true** row max and its matching normalizer (what a real forward
   pass would have saved).
2. A **shifted** but still mathematically valid pair: `m` offset upward by
   a random positive amount, with `l` recomputed to match *that* `m`
   (softmax is shift-invariant, so the correct `P`/`O` are identical either
   way). This catches implementations that silently ignore the supplied
   `m`/`l` and pair a freshly recomputed max with the caller's `l` (or vice
   versa), producing incorrect results whenever the supplied `m` isn't the
   row's literal maximum.

For every case, your `P` rows must each sum to $1$ (within $10^{-5}$) and
both `P` and `O` are compared against the oracle's values with
`max_abs_err`; the maximum over every case and every entry must stay below
$10^{-5}$.
