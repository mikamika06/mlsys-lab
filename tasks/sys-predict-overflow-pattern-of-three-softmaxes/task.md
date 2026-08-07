## Context

Softmax over raw scores $z \in \mathbb{R}^n$ is $\mathrm{softmax}(z)_i =
e^{z_i} / \sum_j e^{z_j}$. Computed **naively** this overflows the
moment any $z_i$ is large enough that $e^{z_i}$ exceeds the float
range (around $z_i \approx 709$ in float64) — even though the true
output probabilities are always finite numbers in $[0,1]$. The standard
fix is **log-sum-exp (LSE) stabilization**: subtract the max first,

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i-m}}{\sum_j e^{z_j-m}}, \qquad m=\max_j z_j,
$$

so every exponent is $\le 0$ and every term is in $(0,1]$ — this never
overflows. When scores arrive one at a time (as in FlashAttention-style
streaming attention), a single stabilizing max isn't known in advance,
so the **online softmax** algorithm (Milakov & Gimelshein) maintains a
running max $m$ and running sum $s$ that both get **rescaled** whenever
a new element raises the max:

$$
m_{\text{new}} = \max(m, z_i), \qquad
s \leftarrow s\cdot e^{\,m - m_{\text{new}}} + e^{\,z_i - m_{\text{new}}}, \qquad
m \leftarrow m_{\text{new}}.
$$

After one pass, the final probabilities are $e^{z_i - m}/s$. Done
correctly, this is exactly as stable as LSE — it just computes the same
answer incrementally, one score at a time.

## Task

Implement `classify_softmax_overflow(z)`:

```python
def classify_softmax_overflow(z: list[float]) -> tuple[bool, bool, bool]:
    ...
```

`z` is a 1-D array of raw scores (values may be as large in magnitude
as $10^4$). Run all three softmax variants described above on `z`:

1. **naive**: `exp(z) / sum(exp(z))`, no stabilization.
2. **lse**: `exp(z - max(z)) / sum(exp(z - max(z)))`.
3. **online**: the single running-max/running-sum streaming algorithm
   above, scanning `z` once to get the final `(m, s)`, then computing
   `exp(z - m) / s`.

Return `(naive_overflow, lse_overflow, online_overflow)`: three
booleans, each `True` iff that variant's output vector contains at
least one `inf` or `nan`.

## Example

```python
classify_softmax_overflow([10000.0, 0.0, -10000.0])
# naive:  exp(10000) overflows to inf -> True
# lse:    max-subtracted, every exponent <= 0            -> False
# online: correctly rescaled running sum, same as lse     -> False
# (True, False, False)
```

## What the gate checks

The grader runs a handful of fixed extreme cases (scores up to $10^4$
in magnitude, including all-equal and all-very-negative vectors) plus 8
randomly scaled vectors (seeded generator, scale factors
from 1 up to 10000) through an independent oracle that implements the
same three algorithms directly and checks `math.isfinite` on each output.
`exact_match` requires all three booleans to match on every case. A
common bug — implementing "online" as a naive running sum of raw
`exp(z_i)` without ever rescaling by `exp(old_m - new_m)` when the max
updates — behaves identically to the naive algorithm and will report
`online_overflow=True` on the extreme cases where the correct algorithm
(matching `lse`) reports `False`.
