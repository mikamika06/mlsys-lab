## Context

Structured sparsity removes weights from a tensor while keeping a predictable pattern that hardware kernels can exploit. In an $N:M$ sparsity pattern, weights are divided into groups of $M$ consecutive values and exactly $N$ values with the largest magnitudes are kept in each group.

For a group $g = (w_1, \dots, w_M)$, the mask keeps indices corresponding to

$$
\operatorname{TopN}(|g|) ,
$$

where $|g|$ is the elementwise absolute value. The remaining weights are pruned.

A binary mask $m$ is applied as

$$
\hat{w}_i = m_i w_i ,
$$

where $m_i \in \{0,1\}$. Every complete group must satisfy

$$
\sum_{i=1}^{M} m_i = N .
$$

This task generalizes common patterns such as $2:4$ sparsity to arbitrary valid $N:M$ configurations.

## Task

Implement `nm_mask(weights, N, M)`:

```python
def nm_mask(weights: np.ndarray, N: int, M: int) -> np.ndarray:
    ...
```

The function receives a one-dimensional NumPy array of floating point weights and positive integers $N$ and $M$. The length of `weights` is guaranteed to be divisible by $M$, and $0 < N \le M$.

Return an integer NumPy array of the same shape where each element is $1` for a surviving weight and `0` for a pruned weight. In every group of $M$ consecutive weights, exactly $N$ entries must be selected.

Selection must match the oracle rule: keep the $N$ largest values by absolute magnitude inside each group. If magnitudes are tied, choose the lower index first.

## Example

```python
import numpy as np

weights = np.array([0.1, -3.0, 2.0, 0.5, 8.0, -1.0, 4.0, 2.0])
mask = nm_mask(weights, 2, 4)

# mask:
# [0, 1, 1, 0, 1, 0, 1, 0]
```

The first group keeps $-3.0$ and $2.0$. The second group keeps $8.0$ and $4.0$.

## What the gate checks

The gate builds several weight arrays and computes the reference mask using a NumPy oracle that ranks each $M$-element group by descending absolute value with stable index ordering.

The returned mask must exactly match the oracle. The gate also verifies that every $M$-element group contains exactly $N$ survivors and that the total absolute magnitude of removed weights matches the oracle result.
