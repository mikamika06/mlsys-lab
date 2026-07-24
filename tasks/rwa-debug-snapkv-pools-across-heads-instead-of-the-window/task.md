## Context

SnapKV selects important tokens from an observation window of attention scores. The
selection is based on aggregating attention over the observation-window token
axis. For an attention tensor with shape $(h, w)$, where $h$ is the number of
heads and $w$ is the number of tokens in the observation window, the token score
for position $j$ is computed by averaging across heads:

$$s_j = \frac{1}{h}\sum_{i=1}^{h} A_{ij}.$$

The selected token positions are the indices of the largest scores. The pooling
axis matters: averaging across the token axis would produce one value per head,
which is not a token importance score.

For a requested budget $k$, the output is the set of $k$ token indices with the
highest values in $s$. Ties are resolved by choosing smaller indices first.

## Task

Implement `select_snapkv_indices(attn, k)`:

```python
def select_snapkv_indices(attn: np.ndarray, k: int) -> np.ndarray:
    ...
```

The input `attn` is a 2-D NumPy array of shape $(h, w)$ containing attention
scores for $h$ attention heads over $w$ observation-window tokens.

Return a 1-D NumPy array containing exactly $k$ integer token indices. The indices
must be sorted in ascending order in the returned array.

The implementation must pool over the head axis to create one score per token.
Do not pool over the token axis.

## Example

```python
import numpy as np

attn = np.array([
    [0.1, 0.8, 0.2, 0.4],
    [0.3, 0.6, 0.9, 0.2],
])

idx = select_snapkv_indices(attn, 2)
# The token scores are [0.2, 0.7, 0.55, 0.3]
# The two largest positions are 1 and 2.
# idx == array([1, 2])
```

## What the gate checks

The gate builds several attention matrices and computes the expected selected
token indices with a NumPy oracle. It requires the implementation output to
exactly match the oracle result. Implementations that average over the
observation-window token axis instead of the head axis produce head scores rather
than token scores and fail the check.
