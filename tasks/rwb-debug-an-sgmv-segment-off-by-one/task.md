## Context

Segmented matrix-vector multiplication (SGMV) is used in adapter systems where
different rows of a batch use different adapter weights. A batch of input rows
$X \in \mathbb{R}^{n \times d}$ is split into segments. Each segment selects an
adapter matrix $W_k \in \mathbb{R}^{d \times m}$, and rows in that segment are
multiplied by the same adapter.

For row $i$, the correct output is

$$
Y_i = X_i W_{s_i},
$$

where $s_i$ is the adapter index assigned to row $i$. The segment metadata gives
the start and end row positions for each adapter. A one-row offset error in these
positions applies the wrong adapter to some rows while still producing valid
matrix multiplications.

## Task

Implement `sgmv(X, adapters, segments)`:

```python
def sgmv(X: np.ndarray, adapters: list[np.ndarray], segments: list[tuple[int, int, int]]) -> np.ndarray:
    ...
```

Arguments:

- `X` is a 2-D `float64` NumPy array with shape $(n, d)$.
- `adapters` is a list where `adapters[k]` has shape $(d, m)$.
- `segments` contains tuples `(start, end, adapter_id)`. Rows in the half-open
  interval $[start, end)$ must use `adapters[adapter_id]`.

Return a `float64` array `Y` with shape $(n, m)$ containing the per-row adapter
products. Do not assume segments have equal length.

## Example

```python
import numpy as np

X = np.array([[1., 2.], [3., 4.], [5., 6.]])
adapters = [
    np.array([[1., 0.], [0., 1.]]),
    np.array([[2., 0.], [0., 2.]])
]
segments = [(0, 2, 0), (2, 3, 1)]

Y = sgmv(X, adapters, segments)
# [[1., 2.],
#  [3., 4.],
#  [10., 12.]]
```

## What the gate checks

The grader builds a NumPy reference implementation that applies each segment's
adapter to exactly the rows in its half-open interval. The returned matrix is
compared with the reference using maximum absolute error:

$$
\max_{i,j} |Y_{ij} - Y^{\mathrm{ref}}_{ij}|.
$$

The error must be less than $10^{-5}$. A shifted segment start or end index
causes rows to receive the wrong adapter and fails the gate.
