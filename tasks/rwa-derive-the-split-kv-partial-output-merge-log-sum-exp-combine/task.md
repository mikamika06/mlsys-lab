## Context

Production attention kernels often split the key-value sequence across multiple workers. Each worker computes a partial softmax result over its own range of keys. The partial results cannot be averaged directly because each range has a different softmax normalization.

For a query vector $q$ and key vectors $K_i$, attention logits are

$$
s_i = q^\top K_i .
$$

The softmax output is

$$
\mathrm{Attn}(q,K,V) =
\frac{\sum_i \exp(s_i)V_i}{\sum_i \exp(s_i)} .
$$

To make split computation numerically stable, each KV range returns a triple $(m_i,l_i,o_i)$ where

$$
m_i = \max_j(s_j),
$$

$$
l_i = \sum_j \exp(s_j-m_i),
$$

and

$$
o_i = \sum_j \exp(s_j-m_i)V_j .
$$

The partial outputs can be merged by choosing

$$
m = \max_i(m_i),
$$

then combining the normalization and output terms:

$$
l = \sum_i \exp(m_i-m)l_i ,
$$

$$
o = \frac{\sum_i \exp(m_i-m)o_i}{l}.
$$

This is the same log-sum-exp correction used by split-KV attention implementations.

## Task

Implement `merge_split_kv(partials)`:

```python
def merge_split_kv(partials):
    ...
```

`partials` is a list of tuples `(m_i, l_i, o_i)` from independent KV ranges.

- `m_i` is a scalar float.
- `l_i` is a scalar float containing the local softmax denominator.
- `o_i` is a 1-D NumPy array containing the local unnormalized output vector.

Return the final attention output as a NumPy array.

The implementation must use the log-sum-exp merge rule. Do not assume the partial ranges have equal sizes or similar logit magnitudes.

## Example

```python
import numpy as np

partials = [
    (2.0, 1.5, np.array([3.0, 6.0])),
    (3.0, 2.0, np.array([4.0, 8.0])),
]

out = merge_split_kv(partials)
```

The returned vector is the normalized combination of both partial attention ranges.

## What the gate checks

The gate builds multiple independent KV ranges, computes their partial triples, and compares the merged output against a NumPy float64 single-pass attention oracle.

The reported metric is `max_abs_err`, the maximum absolute difference between the submitted output and the oracle output. The merge must satisfy

$$
\max_j |x_j-\hat{x}_j| < 10^{-5}.
$$

A solution that averages partial outputs, ignores the different maxima, or uses incorrect normalization will fail.
