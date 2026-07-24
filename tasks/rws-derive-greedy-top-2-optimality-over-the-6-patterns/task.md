## Context

Structured pruning for neural network compression often uses the $2:4$ sparsity pattern. Each group of four weights keeps exactly two values and drops two values.

For a weight group $w = (w_1,w_2,w_3,w_4)$, a valid pattern chooses two indices to keep. There are

$$
\binom{4}{2} = 6
$$

possible keep patterns. The quality of a pattern can be measured by the total magnitude of the weights that are dropped:

$$
L(S) = \sum_{i \notin S} |w_i|,
$$

where $S$ is the set of two kept indices.

A brute-force implementation evaluates all six possibilities. The optimal solution keeps the two largest-magnitude weights because minimizing the dropped magnitude is equivalent to maximizing the kept magnitude:

$$
\arg\min_S L(S) = \text{indices of the two largest } |w_i|.
$$

Production pruning libraries use this greedy rule instead of enumerating all six cases.

## Task

Implement `greedy_24_prune(W)`:

```python
def greedy_24_prune(W: np.ndarray) -> np.ndarray:
    ...
```

`W` is a NumPy array with shape `(n, 4)`. Each row is one independent 2:4 pruning group.

Return a NumPy array of shape `(n,)` containing the minimum dropped magnitude for each group. The returned values must be `float64`.

For every row, compute the sum of the two smallest absolute values. Do not modify `W`.

## Example

```python
import numpy as np

W = np.array([
    [1.0, -4.0, 2.0, 8.0],
    [-3.0, 0.5, 0.25, 2.0],
])

out = greedy_24_prune(W)
# array([3. , 0.75])
```

## What the gate checks

The gate computes a brute-force oracle by evaluating all six valid keep patterns for every input row. The returned vector must exactly match the oracle within numerical tolerance.

The gate also verifies that the result is the same as the greedy top-2 derivation: keeping the two largest values by magnitude leaves the two smallest magnitudes to be dropped.
