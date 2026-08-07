## Context

llama.cpp-style k-quants don't just round each block to a scale derived
from its max magnitude — they **search** a small grid of candidate
scales and keep whichever one minimizes the *importance-weighted*
reconstruction error, using a per-weight importance vector $w$ (an
"imatrix", capturing how much each weight actually matters to the
model's output, e.g. from calibration-data activation statistics).
For a block of values $x \in \mathbb{R}^n$ with weights $w \in
\mathbb{R}^n_{>0}$, integer codes are restricted to $q_i \in
[-n_{\max}, n_{\max}]$, and a candidate scale $d$'s codes and
error are:

$$
q_i(d) = \mathrm{clip}\big(\mathrm{round}(x_i / d),\, -n_{\max},\, n_{\max}\big), \qquad
\mathrm{err}(d) = \sum_i w_i \big(x_i - d \cdot q_i(d)\big)^2
$$

The candidate grid is built around the natural base scale
$d_0 = \max_i |x_i| / n_{\max}$ (the scale that lets the largest-magnitude
element reach exactly $n_{\max}$), nudged by 31 evenly-spaced multipliers:

$$
d_k = d_0 \cdot \left(1 + \frac{k}{32}\right), \qquad k = -15, -14, \dots, 15
$$

(indexed $0$ to $30$ via $\text{idx} = k + 15$). The chosen scale is
whichever $d_k$ minimizes $\mathrm{err}(d_k)$ — the *unweighted* rounding
choice (`k=0`, plain round-to-nearest at the natural scale) is rarely
optimal once importance weights are unequal, since a small nudge to the
scale can trade a larger error on an unimportant element for a smaller
error on an important one.

## Task

Implement `make_qx_quants`:

```python
def make_qx_quants(x: list[float], w: list[float], nmax: int) -> tuple[int, list[int]]:
    ...
```

- `x`: `(n,)` float64 block of values.
- `w`: `(n,)` float64 positive importance weights.
- `nmax`: positive int; codes must lie in `[-nmax, nmax]`.

If every element of `x` is exactly `0`, return `(-1, zeros)` (an all-zero
block has no meaningful scale; codes are all `0`). Otherwise, build the
31 candidate scales $d_k$ as defined above (in order $k=-15,\dots,15$,
so `idx = k + 15` ranges `0..30`), compute each candidate's codes and
importance-weighted error, and return `(best_idx, best_codes)`: the
index of the **first** candidate achieving the minimum error (ties go to
the smaller index, i.e. plain `argmin` over the 31 errors in order), and
that candidate's integer code array (shape `(n,)`).

## Example

```python

x = [1.0, -2.0, 4.0, 0.5]
w = [0.1, 0.1, 10.0, 0.1]  # element 2 matters far more than the rest
make_qx_quants(x, w, nmax=7)
# d0 = max(|x|)/7 = 4/7. Plain round-to-nearest (k=0) may quantize the
# heavily-weighted element 2 slightly off; the search should find a k
# whose scale reconstructs x[2] especially well, even if that costs a
# little extra error on the low-weight elements.
```

## What the gate checks

The grader builds several `(x, w, nmax)` blocks from a seeded Python
generator — uniform weights (search should mostly agree with plain
rounding), sharply skewed weights concentrating importance on one or two
elements, an all-zero block, and different `nmax` values (matching q4-
and q8-style code ranges) — and computes the reference `(best_idx,
best_codes)` independently in Python using the exact 31-candidate grid and
error formula above, never calling your function or hardcoding an
expected value.

`argmin_index` is the fraction of scenarios where **both** your
`best_idx` and your `best_codes` array match the oracle's exactly, and
the gate requires `1.0`. Searching without the importance weights
(minimizing plain squared error instead of the weighted sum), using a
different candidate grid or step size, breaking ties toward the *last*
minimum instead of the first, or forgetting to clip codes to
`[-nmax, nmax]` will all diverge from the oracle on at least one
scenario.
