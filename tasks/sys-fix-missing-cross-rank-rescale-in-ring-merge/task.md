## Context

Ring attention splits a long sequence across ranks. Each rank computes partial softmax statistics for its local block, then ranks merge those statistics into one output.

For a query row with logits $s$, attention output is

$$
\mathrm{Attn}(s,V)=\frac{\sum_j e^{s_j-m}V_j}{\sum_j e^{s_j-m}},
$$

where $m=\max_j s_j$ is used for numerical stability.

A rank only sees its own slice of the logits, so it can only shift by its own
local maximum $m_r$. What it ships to the merge is

$$
l_r=\sum_j e^{s_j-m_r}, \qquad
a_r=\sum_j e^{s_j-m_r}V_j .
$$

Every rank's numbers are therefore expressed on a different scale, and the merge
has to put them on one scale before it may add them. The shipped implementation
below adds them as they arrive.

## Task

Implement `ring_merge(partials)`:

```python
def ring_merge(partials):
    ...
```

`partials` is a list of tuples `(m, l, a)` from different ring ranks.

Each item contains:

- `m`: a NumPy array of shape $(n,)$ containing the local maximum logits.
- `l`: a NumPy array of shape $(n,)$ containing the local exponential sums.
- `a`: a NumPy array of shape $(n, d)$ containing the local weighted value sums.

Return the merged attention output as a NumPy array of shape $(n, d)$ with dtype `float64`.

The result must equal what you would get by concatenating every rank's logits
and values first and taking one softmax over the whole row.

## Example

```python
import numpy as np

partials = [
    (
        np.array([2.0]),
        np.array([1.5]),
        np.array([[3.0, 1.0]])
    ),
    (
        np.array([5.0]),
        np.array([2.0]),
        np.array([[4.0, 6.0]])
    ),
]

out = ring_merge(partials)
```

The result should represent the same output as if both ranks' logits and values had been concatenated before applying softmax.

## What the gate checks

The gate creates synthetic query blocks and values, computes the true attention output with a NumPy softmax oracle, converts each rank block into local $(m_r,l_r,a_r)$ statistics, and compares `ring_merge` against the oracle.

The relative error

$$
\mathrm{rel\_err} =
\frac{\lVert x-\hat{x}\rVert_2}{\lVert x\rVert_2+10^{-12}}
$$

must be at most $10^{-5}$. The cases deliberately give the ranks logits drawn
from different means, so any merge that treats the per-rank numbers as already
comparable is wrong by a factor that grows with the spread between $m_r$ values.
