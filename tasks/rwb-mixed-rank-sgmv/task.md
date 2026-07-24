## Context

A batched adapter system may apply a different low-rank update to each input row. This pattern appears in serving systems where requests from multiple users use different adapters in one batch.

A low-rank adapter update with rank $r_i$ transforms an input vector $x_i \in \mathbb{R}^{d}$ using two matrices:

$$
y_i = x_i + B_i(A_i x_i),
$$

where $A_i \in \mathbb{R}^{r_i \times d}$ and $B_i \in \mathbb{R}^{d \times r_i}$. The adapter rank $r_i$ can vary between rows, so the batch contains ragged matrices rather than one stacked tensor.

A production implementation avoids padding every adapter to the largest rank. Instead, it uses the adapter id of each row to select the correct pair of matrices and applies only that adapter's rank.

## Task

Implement `mixed_rank_sgmv`:

```python
def mixed_rank_sgmv(
    x: np.ndarray,
    adapter_ids: np.ndarray,
    adapters: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    ...
```

Inputs:

- `x` is a `float64` array of shape $(n, d)$ containing input rows.
- `adapter_ids` is an integer array of shape $(n,)$ selecting which adapter is used by each row.
- `adapters` is a list where entry `k` is `(A_k, B_k)`. Each adapter may have a different rank:
  - $A_k$ has shape $(r_k, d)$.
  - $B_k$ has shape $(d, r_k)$.

Return an array of shape $(n, d)$ where each row is:

$$
y_i = x_i + B_{a_i}(A_{a_i}x_i),
$$

and $a_i$ is the adapter id for row $i`.

The implementation should correctly handle adapters with different ranks. It is acceptable to group rows internally, but the returned rows must preserve the original order.

## Example

```python
import numpy as np

x = np.array([[1.0, 2.0], [3.0, 4.0]])
adapter_ids = np.array([0, 1])

adapters = [
    (np.array([[1.0, 0.0]]), np.array([[1.0], [0.0]])),
    (np.array([[0.0, 1.0], [1.0, 0.0]]), np.eye(2)),
]

y = mixed_rank_sgmv(x, adapter_ids, adapters)
```

The first row uses a rank-$1$ adapter and the second row uses a rank-$2$ adapter.

## What the gate checks

The gate builds several batches containing adapters with different ranks. It computes the reference result by applying the selected adapter to each row using a NumPy float64 oracle.

The returned matrix is compared with the oracle using the maximum absolute error:

$$
\max_{i,j} |y_{ij}^{candidate} - y_{ij}^{reference}|.
$$

The gate passes when this value is below $10^{-5}$.
