## Context

Single-head **scaled dot-product attention** (SDPA) turns three matrices
$Q, K, V \in \mathbb{R}^{S \times d}$ (the query, key and value rows for a length-$S$
sequence with head dimension $d$) into an output $O \in \mathbb{R}^{S \times d}$:

$$
\text{scores}_{ij} = \frac{1}{\sqrt{d}}\, Q_i \cdot K_j
= \frac{1}{\sqrt{d}} \sum_{t=1}^{d} Q_{it} K_{jt},
\qquad
P_{i:} = \operatorname{softmax}(\text{scores}_{i:}),
\qquad
O_{it} = \sum_{j=1}^{S} P_{ij}\, V_{jt}.
$$

The usual one-liner leans on BLAS: `P = softmax(Q @ K.T / sqrt(d)); O = P @ V`.
This exercise asks you to build the same result **from scratch** — every dot
product spelled out as an explicit Python loop, with **no** `@`, `np.matmul`,
`np.dot`, `np.einsum`, `np.tensordot` or `np.inner` anywhere. The softmax must be
numerically stable (subtract the per-row max before exponentiating).

Doing the two contractions by hand costs on the order of $2\,S^2 d$
multiply-accumulate steps — one $S^2 d$ pass for the scores and one for the
$P V$ product.

## Task

Implement `sdpa`:

```python
def sdpa(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    ...
```

- `Q`, `K`, `V` are 2-D `float64` arrays of shape $(S, d)$.
- Return the $(S, d)$ attention output.
- Compute **both** the $Q K^\top$ scores and the $P V$ output with explicit
  nested `for` loops over the sequence and head dimensions. Do not call any
  matrix-multiply / dot / einsum primitive — the grader bans them at runtime and
  in your source.
- Use a max-shifted softmax for stability.

## Example

```python
import numpy as np
Q = np.array([[1.0, 0.0], [0.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = np.array([[10.0, 0.0], [0.0, 20.0]])
O = sdpa(Q, K, V)
# scores = [[1/sqrt2, 0], [0, 1/sqrt2]]  ->  softmax gives ~[0.67, 0.33] per row
# O ~= [[6.70, 6.58], [3.30, 13.42]]
```

## What the gate checks

Two things, both measured against a real oracle (never hardcoded):

1. **`max_abs_err`** — the largest absolute difference between your output and a
   vectorized NumPy SDPA reference, over several random shapes. Must be
   $\le 10^{-5}$.

2. **`op_ratio`** — the number of Python line-events your call emits (recorded
   with `sys.settrace`), divided by the line-events of a genuine from-scratch
   triple-loop reference on the same input. Must land in $[0.5, 2.0]$. A
   vectorized or half-vectorized solution runs its arithmetic in C, emits far
   too few line-events, and falls below $0.5$; wildly over-nested work overshoots
   $2.0$. Only an honest $O(S^2 d)$ loop implementation sits near $1.0$.

Any use of `@`, `np.matmul`, `np.dot`, `np.einsum`, `np.tensordot` or `np.inner`
(caught in your source or by a runtime trap that raises) fails the correctness
gate outright.
