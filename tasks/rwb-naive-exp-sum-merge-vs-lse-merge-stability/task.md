## Context

FlashAttention, ring attention, and any scheme that splits attention over
key/value **chunks** (different blocks, different devices, different KV
pages) all face the same problem: each chunk only sees its own local
scores, but the final result must be the softmax over ALL chunks combined.
Given $C$ chunks of raw scores $L_1, \dots, L_C$ and matching value blocks
$V_1, \dots, V_C$, the target quantity is always

$$
\text{output} = \frac{\sum_{i=1}^{C} \sum_j e^{L_i[j]} V_i[j]}{\sum_{i=1}^{C} \sum_j e^{L_i[j]}} .
$$

Computed literally (**naive merge**), this overflows the moment any score
gets much above $\approx 709$ (where $e^{709}$ already exceeds float64's
max representable value $\approx 1.8\times 10^{308}$): both the numerator
and denominator become $+\infty$, and $\infty / \infty$ is `NaN`.

The fix (**LSE merge**, exactly what FlashAttention's block-merge step
does) is to stabilize *locally* first — for chunk $i$, subtract its own
max $m_i = \max(L_i)$ before exponentiating — and then combine the
locally-stabilized chunks by rescaling every one of them to a single
*global* max $g = \max_i(m_i)$:

$$
\alpha_i = e^{m_i - g} \; (\le 1, \text{never overflows}), \qquad
\text{output} = \frac{\sum_i \alpha_i \left(\sum_j e^{L_i[j]-m_i} V_i[j]\right)}{\sum_i \alpha_i \left(\sum_j e^{L_i[j]-m_i}\right)} .
$$

This is mathematically the *same* quantity as the naive formula — only the
order of operations changes — but it never exponentiates anything that
can overflow.

## Task

Implement both `naive_merge` and `lse_merge`:

```python
def naive_merge(chunk_scores: list[np.ndarray], chunk_values: list[np.ndarray]) -> np.ndarray: ...
def lse_merge(chunk_scores: list[np.ndarray], chunk_values: list[np.ndarray]) -> np.ndarray: ...
```

* `chunk_scores` — list of $C$ 1-D arrays; `chunk_scores[i]` has length
  $n_i$.
* `chunk_values` — list of $C$ 2-D arrays; `chunk_values[i]` has shape
  $(n_i, d)$.

`naive_merge` must compute the target quantity **exactly as written above**
— sum `exp(L_i[j])` directly, with **no max-subtraction anywhere**. It is
*supposed* to be unstable; do not "fix" it.

`lse_merge` must compute the identical target quantity via the two-stage
stabilization described above: locally stabilize each chunk with its own
max, then merge the $C$ chunks with a global-max rescale. Both functions
return a `(d,)` vector.

## Example

```python
import numpy as np

chunk_scores = [np.array([800.0, 810.0]), np.array([805.0])]
chunk_values = [np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[2.0, 2.0]])]

naive_merge(chunk_scores, chunk_values)  # -> [nan, nan]  (exp(800) already overflows)
lse_merge(chunk_scores, chunk_values)    # -> a finite, correctly-weighted (2,) vector
```

## What the gate checks

- **lse_err** — across an ordinary-magnitude case and a huge-magnitude
  case (scores up to ~950), your `lse_merge` output must match an
  independently computed stable reference to `<= 1e-5`.
- **naive_check** — your `naive_merge` must match the reference's naive
  computation on BOTH cases: closely (`<= 1e-6`) on the ordinary-magnitude
  case (where naive is still accurate), and exactly `NaN` in every
  component on the huge-magnitude case (where the reference's own literal
  formula also overflows to `NaN`) — reproducing the *same* instability,
  not dodging it.

Both gates must pass.
