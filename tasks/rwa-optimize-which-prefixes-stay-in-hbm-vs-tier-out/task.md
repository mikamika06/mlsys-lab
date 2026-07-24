## Context

Large language model serving systems often keep frequently reused prefixes in high bandwidth memory (HBM) while moving less valuable prefixes to a slower tier. The HBM space is limited, so the system should select prefixes with the highest reuse value per byte.

For a prefix $i$, define its reuse value as

$$v_i = f_i \cdot l_i,$$

where $f_i$ is the reuse frequency and $l_i$ is the prefix length. If the prefix consumes $b_i$ bytes in HBM, its value density is

$$d_i = \frac{v_i}{b_i} = \frac{f_i l_i}{b_i}.$$

The production policy used in this task is a density-based greedy allocator: sort prefixes by decreasing $d_i$ and keep prefixes while the HBM budget allows. The resulting hit rate is the fraction of total reuse value retained:

$$\mathrm{hit\_rate} =
\frac{\sum_{i \in K} v_i}{\sum_i v_i},$$

where $K$ is the set of prefixes kept in HBM.

## Task

Implement `optimize_hbm_prefixes(prefixes, budget)`.

`prefixes` is a list of tuples:

```python
(index, reuse_freq, length, bytes_used)
```

where `index` is the prefix identifier, `reuse_freq` is the observed reuse count, `length` is the prefix token length, and `bytes_used` is the HBM memory required.

Return a tuple:

```python
(kept_indices, hit_rate)
```

where:

- `kept_indices` is the list of selected prefix indices in the order chosen by the density policy.
- `hit_rate` is the retained reuse-value fraction as a Python float.

Use the greedy density algorithm described in the context. Do not mutate the input list.

## Example

```python
prefixes = [
    (10, 100, 20, 1000),
    (11, 20, 50, 100),
    (12, 10, 10, 500),
]

kept, rate = optimize_hbm_prefixes(prefixes, 600)

# Prefix 11 has the highest value density and fits.
# Prefix 10 is next by density but does not fit.
# Prefix 12 is considered last and does not fit.
# kept == [11]
```

## What the gate checks

The gate computes the reference result using the density calculation and greedy allocation algorithm independently for several prefix sets.

The returned kept prefix identifiers must exactly match the oracle selection. The returned hit rate must also exactly match the oracle value.
