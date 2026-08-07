## Context

When attention is split across ranks, each rank sees only its own slice of the
key/value sequence. All it can ship back is a three-part summary of that slice,
per query row:

$$
m_r = \max_j s_{r,j}, \qquad
l_r = \sum_j e^{s_{r,j}-m_r}, \qquad
o_r = \sum_j e^{s_{r,j}-m_r} v_{r,j} .
$$

The raw logits and the values are gone. That summary is deliberately tiny —
$O(d_v)$ per query row instead of $O(N)$ — and the question is how much you can
still recover from it.

More than the output. The same three numbers also carry the **global
log-sum-exp** of the row, which the backward pass needs and which no single rank
can compute, and the **share of the softmax mass each rank contributed**, which
is what you look at when one rank's slice turns out to dominate every row and
you want to know whether the context split is buying you anything.

Both fall out of the same rescaling that produces the output, and both are wrong
in the same way if you skip it.

## Task

Implement `reconstruct_output(states)`:

```python
def reconstruct_output(states):
    ...
```

`states` is a non-empty list of tuples `(m, l, o)`, one per rank, in rank order:

- `m` — shape $(n,)$, the per-query local maxima.
- `l` — shape $(n,)$, the per-query local exponential sums.
- `o` — shape $(n, d_v)$, the per-query local weighted value sums.

Ranks do not hold equal numbers of keys, and their logits do not share a range.

Return a tuple of three `float64` list:

1. `output` — shape $(n, d_v)$: the attention output, identical to concatenating
   every rank's logits and values and taking one softmax over the whole row.
2. `global_lse` — shape $(n,)$: $\log \sum_j e^{s_j}$ over **all** keys of the
   row, on the original logit scale, not shifted by any local maximum.
3. `rank_mass` — shape $(R, n)$ where $R = \texttt{len(states)}$: entry $(r, i)$
   is the fraction of query row $i$'s total softmax mass contributed by rank
   $r$. Every column sums to $1$.

## Example

```python

states = [
    ([2.0], [1.5], [[3.0, 6.0]]),
    ([1.0], [2.0], [[4.0, 8.0]]),
]

output, global_lse, rank_mass = reconstruct_output(states)
```

Rank $0$ carries $1.5\,e^{2}$ of the mass and rank $1$ carries $2.0\,e^{1}$, so
`rank_mass[:, 0]` is about `[0.6709, 0.3291]` — the first rank dominates even
though it shipped the smaller $l$.

## What the gate checks

Three metrics, all against an oracle that rebuilds the full logit matrix in
Python and never touches the per-rank summaries.

`max_abs_err` — the output. Below $10^{-6}$.

`lse_abs_err` — `global_lse`. Below $10^{-8}$. Returning the shifted quantity
$\log \sum_r l_r e^{m_r-M}$ without adding $M$ back scores in the hundreds on
the wide-spread case, so this metric catches exactly the step that is easy to
drop.

`mass_abs_err` — `rank_mass`. Below $10^{-9}$. A matrix transposed to $(n, R)$
fails on shape; columns that do not sum to $1$ fail on value.

The cases include ranks whose local maxima sit hundreds apart, a rank holding a
single key beside a rank holding eight, and a case where the dominant rank is
not the one with the largest $l$. Any of the three results computed without the
$e^{m_r-M}$ rescale fails at least one metric.
