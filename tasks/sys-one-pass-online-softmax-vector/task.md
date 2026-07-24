## Context

Flash-attention-style kernels never materialize a full attention row: they
stream key/value blocks and keep just three running scalars/vectors — a
running max $m$, a running exponential-sum $l$, and a running
value-weighted accumulator $o$ — updating them one block at a time. At the
end, $o / l$ equals the exact value-weighted softmax output, without ever
computing $\exp$ over the whole $N$-length score vector at once.

For a query's raw scores $x \in \mathbb{R}^N$ against $N$ keys and matching
values $V \in \mathbb{R}^{N \times d}$, the quantity being computed is

$$
\mathrm{out} = \sum_{i=1}^{N} \mathrm{softmax}(x)_i \, V_i
= \frac{\sum_i e^{x_i - m} V_i}{\sum_i e^{x_i - m}}, \qquad m = \max_i x_i .
$$

## Task

Implement `online_softmax_weighted_sum`, processing `scores` and `V` in
contiguous blocks of size `block_size` (the last block may be shorter):

```python
def online_softmax_weighted_sum(scores: np.ndarray, V: np.ndarray, block_size: int) -> np.ndarray:
    ...
```

* `scores` — 1-D array of shape $(N,)$, raw (unnormalized) attention scores.
* `V` — 2-D array of shape $(N, d)$, value vectors.
* `block_size` — block size $B$ to consume `scores`/`V` in.

Maintain running state $(m, l, o)$, initialized to $(-\infty, 0, \vec 0)$.
For each block $c$ (scores $x_c$, values $V_c$):

$$
m_c = \max(x_c), \qquad m_{\text{new}} = \max(m, m_c)
$$
$$
\alpha = e^{\,m - m_{\text{new}}} \quad (\alpha = 0 \text{ on the first block, since } m=-\infty)
$$
$$
p = e^{\,x_c - m_{\text{new}}}
$$
$$
l \leftarrow l\,\alpha + \sum_i p_i, \qquad
o \leftarrow o\,\alpha + p^\top V_c, \qquad
m \leftarrow m_{\text{new}}
$$

After the last block, return $o / l$ — a vector of shape $(d,)$.

**Your implementation must never call `exp` on an array covering all $N$
scores at once** (only on a single block's scores, length $\le B$); doing so
defeats the point of streaming and is checked at grading time.

## Example

```python
import numpy as np
scores = np.array([1.0, 3.0, -1.0, 0.5, 2.0])
V = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 0.], [0., 2.]])
out = online_softmax_weighted_sum(scores, V, block_size=2)
# equals softmax(scores) @ V, computed two elements at a time
```

## What the gate checks

* **max_abs_err** — the returned vector must match
  $\mathrm{softmax}(\text{scores})^\top V$ (computed directly by the grader)
  to within `1e-5`, across several cases with different `N`, `block_size`
  (including `block_size` that doesn't evenly divide `N`, and large-magnitude
  scores where naive unstabilized exponentials would overflow).
* **blockwise** — during grading, `numpy.exp` is instrumented to record the
  length of every array passed to it. If any call's array has length equal
  to the full `N` (and `N > block_size`), the case is flagged and this gate
  is `0.0`. A solution that computes the whole softmax vector up front and
  merely reads `V` in chunks will fail this gate even if its numbers are
  correct.
