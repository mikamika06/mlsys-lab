## Context

Structured pruning systems often operate on groups of coupled parameters rather than individual tensors. A dependency graph can identify tensors that must be pruned together because changing one parameter structure affects related parameters.

The importance of a coupled group can be measured by combining all weights in the group into a single L2 norm. For a group containing tensors $W_1, W_2, \dots, W_k$, the group importance is

$$
I(G) = \sqrt{\sum_{j=1}^{k} \lVert W_j \rVert_2^2}.
$$

Using only one tensor from a dependency group can produce an incorrect pruning order because it ignores the magnitude of the remaining coupled parameters.

## Task

Implement `rank_groups_by_importance(groups)`.

The input is a list of groups. Each group is a dictionary with:

```python
{
    "id": int,
    "tensors": [numpy arrays]
}
```

Return a list containing all group IDs sorted from highest importance to lowest importance.

For every group, compute importance from all tensors:

$$
I(G) = \sqrt{\sum_{j=1}^{k}\sum_x x^2}.
$$

Sort groups by decreasing importance. If two groups have the same importance, sort by increasing group ID.

## Example

```python
import numpy as np

groups = [
    {
        "id": 10,
        "tensors": [np.array([1.0, 1.0]), np.array([2.0])]
    },
    {
        "id": 20,
        "tensors": [np.array([3.0])]
    }
]

rank_groups_by_importance(groups)
# [20, 10]
```

The first group has importance

$$
\sqrt{1^2 + 1^2 + 2^2} = \sqrt{6},
$$

and the second group has importance

$$
\sqrt{3^2}=3.
$$

## What the gate checks

The gate computes the reference ranking using a NumPy implementation of the group L2 norm algorithm. The returned ordering is compared using Spearman rank correlation $\rho$.

A passing implementation must satisfy

$$
\rho = 1.0.
$$

This means every group must appear in the same relative order as the oracle ranking.
