## Context

Padding every sequence in a batch out to the longest one wastes compute
on pad tokens. An alternative used by production kernels (e.g.
xformers' `BlockDiagonalMask`, and the packing used ahead of varlen
FlashAttention) instead **packs** several variable-length sequences
end-to-end into one long tensor with no padding, and relies on an
attention mask shaped like a block-diagonal matrix to keep the sequences
from attending to each other.

Given $k$ sequences of lengths $\ell_1,\dots,\ell_k$ packed row-wise into
$Q, K, V \in \mathbb{R}^{N\times d}$ with $N=\sum_i \ell_i$, assign each
row $r$ a sequence id $s(r) \in \{1,\dots,k\}$ (the sequence it belongs
to). The block-diagonal attention score is

$$
\text{score}_{i,j} =
\begin{cases}
\dfrac{q_i \cdot k_j}{\sqrt d} & s(i) = s(j) \\[4pt]
-\infty & s(i) \ne s(j)
\end{cases}
\qquad
O_i = \sum_j \operatorname{softmax}(\text{score}_{i,:})_j \; v_j
$$

Row $i$'s softmax is taken only over the rows belonging to its own
sequence — attention is full (non-causal) *within* a sequence and exactly
zero *across* sequences. This is mathematically identical to running
ordinary attention independently on each sequence's slice and
concatenating the results, but the packed form is how it is actually
computed in one batched kernel call.

## Task

Implement `block_diagonal_attention`:

```python
def block_diagonal_attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], seq_lens: list[int]) -> list[list[float]]:
    ...
```

- `Q`, `K`, `V`: `(N, d)`, the packed tensors.
- `seq_lens`: a list of positive ints, the length of each packed sequence
  in order; `sum(seq_lens) == N`.
- Build the block-diagonal mask from `seq_lens`, mask disallowed logits
  with `-inf` **before** softmax, and return the `(N, d)` output.

## Example

```python

# two packed sequences: rows 0-2 are sequence A, rows 3-4 are sequence B
seq_lens = [3, 2]
rng = random.Random(0); Q = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(5)]
rng = random.Random(1); K = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(5)]
rng = random.Random(2); V = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(5)]

out = block_diagonal_attention(Q, K, V, seq_lens)
# out[:3] depends only on Q[:3], K[:3], V[:3]  (sequence A)
# out[3:] depends only on Q[3:], K[3:], V[3:]  (sequence B)
```

## What the gate checks

The grader packs several random-length sequences from a seeded RNG and
compares your output to an oracle that slices `Q`, `K`, `V` back out by
`seq_lens` and runs ordinary dense attention independently on each slice
in Python, then concatenates — never calling your function, never
hardcoding an expected value, and structurally unable to leak
cross-sequence information.

`max_abs_err` is the worst per-case max-abs-error across all cases and
must be `<= 1e-5`. Any row that attends across a sequence boundary — even
slightly — will differ from the independently-computed per-sequence
reference.
