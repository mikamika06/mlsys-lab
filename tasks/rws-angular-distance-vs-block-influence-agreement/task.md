## Context

Layer importance methods often compare different signals to decide which blocks of a model are most affected by a change. One signal can be based on the angular distance between a layer's original and updated states. For vectors $x$ and $y$, the angle is

$$
\theta(x,y) = \arccos\left(\frac{x^\top y}{\lVert x\rVert\lVert y\rVert}\right).
$$

A second signal can measure block influence by the relative size of the state change:

$$
I(x,y) = \frac{\lVert y-x\rVert}{\lVert x\rVert + \epsilon}.
$$

To compare the two importance signals, layers are ranked from highest score to lowest score. The agreement between rankings is measured with Spearman correlation. If $r_i$ and $s_i$ are the rank positions of layer $i$, the correlation is

$$
\rho = 1 - \frac{6\sum_i(r_i-s_i)^2}{n(n^2-1)}.
$$

## Task

Implement `angular_distance_vs_block_influence(states)`.

The input `states` is a list of dictionaries. Each dictionary represents one layer and contains:

```python
{
    "before": np.ndarray,
    "after": np.ndarray
}
```

The arrays are flattened layer states with the same shape.

Return a tuple:

```python
(angle_order, influence_order, spearman)
```

where:

- `angle_order` is a list of layer indices sorted by descending angular distance between `before` and `after`.
- `influence_order` is a list of layer indices sorted by descending relative block influence.
- `spearman` is the Spearman correlation between the two rankings as a Python `float`.

Use NumPy for numerical computation. The angle calculation should clamp the cosine value into $[-1,1]$ before calling $\arccos$.

## Example

```python
import numpy as np

states = [
    {"before": np.array([1.0, 0.0]), "after": np.array([0.8, 0.6])},
    {"before": np.array([1.0, 1.0]), "after": np.array([1.2, 1.1])},
]

angle_order, influence_order, rho = angular_distance_vs_block_influence(states)
```

For every layer, the function returns its ranking position according to the two signals and the agreement value between those rankings.

## What the gate checks

The gate builds several layer-state cases and computes the reference answer with an independent NumPy oracle. It checks that both returned rankings match the oracle exactly and that the returned Spearman correlation differs from the oracle by no more than $10^{-6}$.
