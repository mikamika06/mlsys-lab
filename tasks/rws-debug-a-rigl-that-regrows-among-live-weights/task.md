## Context

RigL-style sparse training keeps a binary mask $m$ over weights $w$. A position is
active when $m_i = 1$ and pruned when $m_i = 0$. During a growth step, new active
positions should be selected from currently pruned positions only.

For a gradient vector $g$, the regrowth score is the gradient magnitude

$$s_i = |g_i|.$$

The candidate set is restricted to zero-mask entries:

$$C = \{i \mid m_i = 0\}.$$

The new connections are the $k$ indices in $C$ with the largest values of $s_i$.
Existing live weights are not candidates, even if their gradients are large.

## Task

Fix `rigl_grow(mask, weights, grads, grow_count)` so that it performs the RigL
growth selection correctly.

The function signature is:

```python
def rigl_grow(mask: list[int], weights: list[float], grads: list[float], grow_count: int) -> list[int]:
    ...
```

Inputs are list of floats of equal length. `mask` contains only
zeros and ones. `weights` are the current weights and `grads` are the current
gradients.

Return a new integer mask. The number of active entries must increase by exactly
`grow_count` unless fewer zero-mask positions exist. Select new entries only from
positions where the input mask is zero, ranking candidates by $|g_i|$.

The `weights` argument is provided because the broken implementation may
incorrectly use $|w_i|$ for ranking. The correct implementation does not use
weight magnitude for growth selection.

## Example

```python

mask = [1, 0, 0, 1, 0]
weights = [0.8, 9.0, -2.0, 0.1, 5.0]
grads = [0.1, 0.3, 0.9, 0.2, 0.4]

new_mask = rigl_grow(mask, weights, grads, 2)
# new_mask is [1, 0, 1, 1, 1]
```

The two new entries are positions $2$ and $4$ because their gradient magnitudes
are the largest among currently zero-mask positions.

## What the gate checks

The gate computes an independent Python oracle. It verifies that the returned
mask exactly matches the oracle mask and that the number of live positions is
conserved according to the requested growth count. Implementations that select
from already-live positions or rank by $|w|$ fail the gate.
