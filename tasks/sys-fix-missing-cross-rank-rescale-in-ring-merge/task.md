## Context

Ring attention splits a long sequence across ranks. Each rank computes partial softmax statistics for its local block, then ranks merge those statistics into one output.

For a query row with logits $s$, attention output is

$$
\mathrm{Attn}(s,V)=\frac{\sum_j e^{s_j-m}V_j}{\sum_j e^{s_j-m}},
$$

where $m=\max_j s_j$ is used for numerical stability.

Each rank computes local values using its own maximum $m_r$:

$$
l_r=\sum_j e^{s_j-m_r}, \qquad
a_r=\sum_j e^{s_j-m_r}V_j .
$$

When combining ranks, the maximum can change. If

$$
m=\max_r m_r,
$$

then every rank contribution must be rescaled:

$$
l=\sum_r l_r e^{m_r-m}, \qquad
a=\sum_r a_r e^{m_r-m}.
$$

The final output is $a/l$. Omitting the factor $e^{m_r-m}$ gives incorrect results when ranks have different logit ranges.

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

The implementation must correctly merge all ranks using the cross-rank rescaling factor.

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

must be at most $10^{-5}$. A merge that forgets the cross-rank rescaling fails on inputs where ranks have different local maximum logits.
