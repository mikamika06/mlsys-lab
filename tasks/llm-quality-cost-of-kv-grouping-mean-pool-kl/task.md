## Context

Multi-head attention (MHA) uses one key head per query head. Grouped-query
attention (GQA) reduces memory usage by sharing key and value heads across
groups of query heads. A simple approximation is to replace each group of key
heads with their mean.

For a query vector $q$ and key matrix $K$, attention probabilities are computed
as

$$
p_j = \frac{\exp(q^\top K_j / \sqrt{d})}
{\sum_l \exp(q^\top K_l / \sqrt{d})}.
$$

The quality cost of grouping can be measured by comparing the original MHA
distribution with the grouped distribution using KL divergence:

$$
D_{\mathrm{KL}}(p \parallel r)
=
\sum_j p_j \log\left(\frac{p_j}{r_j}\right).
$$

The final score is the average KL divergence over all batches, heads, and query
positions.

## Task

Implement `kv_grouping_mean_pool_kl(q, k, group_size)`.

Inputs:

- `q`: a NumPy array of shape $(B, H, T_q, D)$ containing query vectors.
- `k`: a NumPy array of shape $(B, H, T_k, D)$ containing MHA key heads.
- `group_size`: a positive integer dividing $H$.

The function must:

1. Compute full MHA attention distributions using each query head and its
   matching key head.
2. Mean-pool key heads inside each group of `group_size`.
3. Compute grouped attention distributions by using the pooled keys for every
   query head in that group.
4. Return the mean KL divergence as a Python `float`.

Use deterministic NumPy computation in `float64`.

## Example

```python
import numpy as np

q = np.array([[[[1.0, 0.0]]]])
k = np.array([[[[1.0, 0.0]]]])

score = kv_grouping_mean_pool_kl(q, k, 1)
# score == 0.0
```

## What the gate checks

The gate computes the expected value with an independent NumPy oracle. The
returned `mean_kl` value must have absolute error at most $10^{-12}$ from the
oracle result.

The check catches incorrect grouping axes, pooling queries instead of keys,
comparing logits instead of attention probabilities, and incorrect assignment
of pooled key heads to query heads.
