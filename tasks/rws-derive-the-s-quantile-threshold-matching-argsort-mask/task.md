## Context

Magnitude pruning and sparse model compression often keep the largest entries of a
weight tensor. For a flattened weight vector $w \in \mathbb{R}^n$, define the
magnitudes

$$
m_i = |w_i| .
$$

A quantile threshold at level $s$ is

$$
t = \mathrm{quantile}(m, s).
$$

Entries with magnitudes larger than the threshold are always kept. When multiple
entries have the same magnitude as the threshold, a deterministic tie-break is
needed so the result exactly matches a top-$k$ selection. Production systems use
a stable ordering rule, such as the original flat index, to choose which tied
entries remain.

The target keep count is

$$
k = \lceil (1-s)n \rceil .
$$

The reference top-$k$ mask is obtained by sorting magnitudes in descending order
with flat-index order as the tie-break. The quantile threshold and this mask must
describe the same keep-set.

## Task

Implement `quantile_keep_mask(w, s)`:

```python
def quantile_keep_mask(w: np.ndarray, s: float):
    ...
```

The function receives a NumPy array `w` and a quantile level `s` with
$0 \le s < 1$. Flatten the array for selection. Return a tuple:

```python
(threshold, mask)
```

where:

- `threshold` is `np.quantile(np.abs(w), s)` as a Python float.
- `mask` has the same shape as `w` and contains booleans.
- `mask` must keep exactly the top $k$ magnitudes.
- Equal-magnitude ties at the boundary must be resolved by lower flat index first.

The implementation should work for arrays containing many duplicate magnitudes.

## Example

```python
import numpy as np

w = np.array([0.2, -0.9, 0.9, 0.1, 0.9])
threshold, mask = quantile_keep_mask(w, 0.6)

# threshold is the 0.6 quantile of [0.2, 0.9, 0.9, 0.1, 0.9]
# k = ceil((1 - 0.6) * 5) = 2
# The two kept entries are indices 1 and 2 because the tie at magnitude 0.9
# is resolved by flat index.
# mask == [False, True, True, False, False]
```

## What the gate checks

The gate computes a NumPy oracle using an argsort top-$k$ keep-set with magnitude
descending order and flat-index ascending tie-break. It compares both the returned
threshold and boolean mask against the oracle on arrays with repeated magnitudes.

A correct solution receives `exact_match = 1.0`.
