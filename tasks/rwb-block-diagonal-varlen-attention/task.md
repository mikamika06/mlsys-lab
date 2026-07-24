## Context

Production varlen attention kernels (FlashAttention's `varlen` API,
xformers' `BlockDiagonalMask`) pack many variable-length sequences into
one buffer with **no padding**, and describe the packing with a single
cumulative-length array `cu_seqlens` instead of per-sequence tensors:

$$
\text{cu\_seqlens} = [0, \ell_1, \ell_1+\ell_2, \dots, \textstyle\sum_i \ell_i]
$$

Sequence $i$ occupies rows `cu_seqlens[i] : cu_seqlens[i+1]` of the
packed `q, k, v \in \mathbb{R}^{N\times d}`. Attention is block-diagonal:
row $r$ (belonging to sequence $s(r)$) only attends to other rows in the
same sequence:

$$
\text{score}_{i,j} =
\begin{cases}
\dfrac{q_i \cdot k_j}{\sqrt d} & s(i) = s(j) \\[4pt]
-\infty & s(i) \ne s(j)
\end{cases}
\qquad
O_i = \sum_j \operatorname{softmax}(\text{score}_{i,:})_j \; v_j
$$

This is mathematically identical to unpacking each sequence, running
ordinary dense attention on it independently, and concatenating the
results back together — but real batches are packed and read via
`cu_seqlens`, not pre-split.

## Task

Implement `varlen_block_diagonal_attention`:

```python
def varlen_block_diagonal_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                                     cu_seqlens: np.ndarray) -> np.ndarray:
    ...
```

- `q`, `k`, `v`: `(N, d)`, packed.
- `cu_seqlens`: `(n_seqs + 1,)` int, `cu_seqlens[0] == 0`,
  `cu_seqlens[-1] == N`. Sequence `i` occupies rows
  `cu_seqlens[i] : cu_seqlens[i+1]`.
- Derive each row's sequence id from `cu_seqlens`, build the
  block-diagonal mask, mask disallowed logits with `-inf` **before**
  softmax, and return the `(N, d)` output.

## Example

```python
import numpy as np

# three packed sequences of length 2, 1, 3 -> N = 6
cu_seqlens = np.array([0, 2, 3, 6])
q = np.random.default_rng(0).standard_normal((6, 4))
k = np.random.default_rng(1).standard_normal((6, 4))
v = np.random.default_rng(2).standard_normal((6, 4))

out = varlen_block_diagonal_attention(q, k, v, cu_seqlens)
# out[0:2] depends only on rows 0:2 (sequence 0)
# out[2:3] depends only on row 2   (sequence 1, a single token)
# out[3:6] depends only on rows 3:6 (sequence 2)
```

## What the gate checks

The grader loads a committed fixture — a real skewed batch (one
47-token sequence among several short 1-9 token ones, packed together)
— plus several additional seeded synthetic packings, and compares your
output to an oracle that unpacks each segment by `cu_seqlens`, runs
ordinary dense attention independently on each slice in NumPy, and
concatenates — never calling your function, never hardcoding an
expected value, and structurally unable to leak cross-sequence
information.

`max_abs_err` is the worst per-case max-abs-error across all cases and
must be `<= 1e-5`. Deriving segment ids from the wrong end of
`cu_seqlens`, using `side="left"` where `"right"` is needed at a segment
boundary, or letting any row attend past its segment all produce an
output that diverges from the independently-computed per-sequence
reference.
