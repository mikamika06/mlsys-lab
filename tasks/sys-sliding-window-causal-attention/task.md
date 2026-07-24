## Context

Sliding-window causal attention restricts query $i$ to attend only to keys
within a fixed-width local window ending at itself:

$$
\text{allowed}(i,j) = (j \le i) \ \wedge\ (i - j < w),
$$

so query $i$ attends to keys $j \in \{\max(0, i-w+1), \dots, i\}$. The
naive way to compute this is to materialize the full $(n,n)$ boolean mask,
apply it to the full $(n,n)$ score matrix $QK^\top/\sqrt d$, and softmax —
exactly what the reference in this task does (it is provided to you
pre-built, as `fixtures/mask.npy`, for one representative case, and the
grader builds the equivalent mask inline for every other case). That
approach costs $O(n^2)$ memory and $O(n^2 d)$ compute even though each
row only ever needs $w$ keys.

A **tiled** implementation instead partitions the queries into
contiguous blocks and, for each block, slices out *only* the range of
keys any query in that block could possibly attend to — never
materializing the full $n \times n$ mask or score matrix:

For a query tile covering rows $[q_s, q_e)$:

$$
k_{\text{lo}} = \max(0,\ q_s - w + 1), \qquad k_{\text{hi}} = q_e,
$$

$$
K_{\text{tile}} = K[k_{\text{lo}}:k_{\text{hi}}], \qquad
V_{\text{tile}} = V[k_{\text{lo}}:k_{\text{hi}}].
$$

Within the tile, scores are computed only against this local key slice
($\text{tile\_rows} \times (k_{\text{hi}} - k_{\text{lo}})$, not
$n \times n$), a small local mask re-derives the same
$\text{allowed}(i,j)$ condition using the tile's absolute row/column
indices, and softmax + weighted sum with $V_{\text{tile}}$ produce that
tile's output rows. The final $(n,d)$ output, assembled tile by tile,
must be numerically identical (up to floating point) to the full-mask
reference.

## Task

Implement `sliding_window_attention_tiled`:

```python
def sliding_window_attention_tiled(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window: int, block_size: int) -> np.ndarray:
    ...
```

* `Q`, `K`, `V` — `float64` arrays of shape $(n, d)$.
* `window` — the window width $w$ (query $i$ attends to keys
  $\max(0, i-w+1),\dots,i$).
* `block_size` — the query tile size. Process queries in contiguous
  tiles of at most `block_size` rows (the last tile may be shorter).

For every query tile, slice out only the relevant key/value range as
described above, compute local masked-softmax attention restricted to
that slice, and write the tile's output rows. Never build the full
$(n,n)$ mask or the full $(n,n)$ score matrix. Return the assembled
$(n, d)$ output array. Scores are scaled by $1/\sqrt d$ as usual.

## Example

```python
import numpy as np

n, d, w, bs = 6, 4, 3, 2
rng = np.random.default_rng(0)
Q = rng.standard_normal((n, d))
K = rng.standard_normal((n, d))
V = rng.standard_normal((n, d))

out = sliding_window_attention_tiled(Q, K, V, window=w, block_size=bs)
print(out.shape)   # (6, 4)
```

## What the gate checks

**max_abs_err** — for several deterministic `(Q, K, V, window,
block_size)` cases (one of them using the pre-built `fixtures/mask.npy`
as the mask, the rest with masks built inline by the grader using the
same $\text{allowed}(i,j)$ formula), the grader computes the reference
output via full-matrix masked softmax attention and compares it
element-wise against your tiled output. The maximum absolute difference,
across all cases, must be $\le 10^{-5}$.
