## Context

llama.cpp-style GGUF weight formats all quantize a 32-value block down to
4-bit signed codes $c_i \in \{-8,\dots,7\}$ plus one floating-point scale
$d$, so the dequantized value is $\hat x_i = d \cdot c_i$. The formats
differ only in **how $d$ is chosen**:

* **Q4_0** (legacy) picks $d$ directly from the block's magnitude, with no
  search:
  $$
  d_0 = \frac{\max_i |x_i|}{8}, \qquad
  c_i = \operatorname{clip}\!\big(\operatorname{round}(x_i / d_0),\, -8,\, 7\big).
  $$

* **Q4_K** (a k-quant) instead *searches* a small grid of candidate scales
  around $d_0$ and keeps whichever minimizes the **plain** (unweighted)
  squared reconstruction error over the block. The grid is 31 evenly
  spaced multipliers of $d_0$:
  $$
  d_k = d_0 \cdot \Big(1 + \frac{k}{32}\Big), \qquad k = -15, -14, \dots, 15.
  $$
  For a candidate scale $d$, codes are $c_i(d) = \operatorname{clip}(\operatorname{round}(x_i/d), -8, 7)$
  and its unweighted error is $\sum_i (x_i - d\,c_i(d))^2$.

* **Imatrix-Q4_K** runs the *exact same* 31-candidate search, but
  minimizes the **importance-weighted** squared error instead, using a
  per-weight importance vector $w \in \mathbb{R}^{32}_{>0}$ (an "imatrix",
  e.g. derived from calibration-data activation statistics):
  $$
  \operatorname{err}_w(d) = \sum_i w_i \big(x_i - d\,c_i(d)\big)^2 .
  $$

All three variants are then judged the same way: by the **importance-weighted
mean squared error** of their reconstruction against the true weights,
using the real weights $w$:
$$
\operatorname{wmse}(\hat x) = \frac{\sum_i w_i (x_i - \hat x_i)^2}{\sum_i w_i} .
$$

Since Q4_0 does not search at all, and Q4_K searches but ignores $w$,
Imatrix-Q4_K should never do *worse* than the other two by this measure —
it directly optimizes the metric it is judged on.

## Task

Implement:

```python
def compare_q4_variants(x: list[float], w: list[float]) -> tuple:
    ...
```

* `x`: `(32,)` `float64` array — one real weight block (the standard
  ggml block size).
* `w`: `(32,)` `float64` array of positive importance weights.

Return `(errors, best_idx)` where:

* `errors`: a `float64` array of shape `(3,)` holding
  `[wmse(Q4_0), wmse(Q4_K), wmse(Imatrix-Q4_K)]`, each the importance-weighted
  MSE (using `w`, as defined above) of that variant's reconstruction against `x`.
* `best_idx`: a Python `int`, the index (`0`, `1`, or `2`) of the smallest
  entry in `errors` — ties resolved by taking the **smallest** index (plain
  `argmin`).

Ties in the 31-candidate scale search itself (equal error for two
candidate $d_k$) are also resolved by taking the smallest `k` (smallest
`idx = k + 15`).

## Example

```python

rng = random.Random(0)
x = rng.normal(size=32)
w = [1.0] * 32
w[5] = 50.0   # element 5 matters far more than the rest

errors, best_idx = compare_q4_variants(x, w)
print(errors)     # e.g. [0.0421, 0.0398, 0.0117]
print(best_idx)    # 2  -- Imatrix-Q4_K wins because it optimizes for w
```

## What the gate checks

The grader builds several `(x, w)` blocks from a seeded Python generator
(uniform weights, sharply skewed weights, an all-equal-magnitude block,
a wide-range calibration-like block) and computes the reference
`(errors, best_idx)` independently in Python using the exact formulas
above — it never calls your function or hardcodes an expected value.

`rel_err` is the relative L2 error between your `errors` array and the
oracle's `errors` array, stacked across all scenarios; it must be
`<= 1e-6`. `argmin_match` is the fraction of scenarios where your
`best_idx` exactly equals the oracle's `best_idx`; it must be `1.0`.

Using the wrong clip range (e.g. `[-8, 8]` instead of `[-8, 7]`), forgetting
to apply the importance weights in the Imatrix-Q4_K search or in the
final `wmse`, using a different candidate grid, or breaking search ties
toward the *last* minimum instead of the first will all diverge from the
oracle on at least one scenario.
