## Context

Many production neural network compression systems use structured sparsity patterns to reduce memory bandwidth and accelerate matrix multiplication. A common format is 2:4 sparsity, where every consecutive group of four weights keeps exactly two values and removes the other two.

For a row of weights $w \in \mathbb{R}^{n}$, the row is partitioned into groups:

$$
(w_0,w_1,w_2,w_3), (w_4,w_5,w_6,w_7), \dots
$$

For each group of four, the selected entries are the two largest values by magnitude. The resulting binary mask $m$ satisfies:

$$
m_i \in \{0,1\}
$$

and for every group $G$ of four consecutive elements:

$$
\sum_{i \in G} m_i = 2.
$$

A common bug is to select the largest half of the entire row instead of selecting independently inside each group of four. That approach can produce rows where some groups have fewer or more than two selected values.

## Task

Implement `select_2_4_mask(W)`:

```python
def select_2_4_mask(W: np.ndarray) -> np.ndarray:
    ...
```

The input is a 2-D NumPy array of weights with shape $(r, c)$. The number of columns is always divisible by $4$.

Return an integer NumPy array of the same shape. Each group of four consecutive values in every row must contain exactly two ones. The two ones must correspond to the two largest absolute values in that group.

Use NumPy operations. Ties may be resolved by selecting lower indices first.

## Example

```python
import numpy as np

W = np.array([
    [1.0, -5.0, 2.0, 0.5],
    [8.0, 1.0, -7.0, 2.0],
])

mask = select_2_4_mask(W)

# The first row keeps -5 and 2:
# [[0, 1, 1, 0],
#
#  The second row keeps 8 and -7:
#  [1, 0, 1, 0]]
```

## What the gate checks

The gate builds the expected 2:4 mask with a NumPy oracle that independently evaluates every row and every group of four. The returned mask must exactly match the oracle output.

The gate also verifies the structural invariant that every group of four selected entries contains exactly two ones:

$$
\forall G,\quad \sum_{i \in G} m_i = 2.
$$

A solution that selects the global top half of each row will fail because it does not enforce the per-group constraint.
