## Context

Structured pruning methods often rank attention heads or channels before removing
less important groups. A simple magnitude heuristic uses the size of the weights,
while a first-order Taylor approximation uses the estimated loss change caused by
removing a group.

For a head with weights $w$ and gradient values $g$, the group magnitude score is

$$
S_{\mathrm{mag}}(w) = \lVert w \rVert_2 =
\sqrt{\sum_i w_i^2}.
$$

The first-order Taylor saliency score used for pruning is

$$
S_{\mathrm{taylor}}(w,g) = \sum_i |g_i w_i|.
$$

For multiple heads, each head receives one score. Larger scores indicate more
important heads. The rankings should list head indices in descending score order,
with lower indices used to break ties.

Magnitude and Taylor scores measure different properties. Magnitude measures
parameter size, while Taylor saliency incorporates the current gradient signal
from backpropagation.

## Task

Implement `rank_heads_by_importance(weights, grads)`:

```python
def rank_heads_by_importance(weights: list[list[float]], grads: list[list[float]]):
    ...
```

The inputs are two list with identical shape $(h, \dots)$, where the
first dimension indexes attention heads. Compute both scores for every head and
return:

```python
(magnitude_ranking, taylor_ranking)
```

where each ranking is a list of head indices ordered from highest importance to
lowest importance.

Use the full tensor values of each head. The returned rankings must use integer
head indices.

## Example

```python

weights = [
    [[3.0, 0.0]],
    [[1.0, 1.0]],
    [[0.5, 0.5]],
]

grads = [
    [[0.1, 0.1]],
    [[5.0, 5.0]],
    [[0.1, 0.1]],
]

mag_rank, taylor_rank = rank_heads_by_importance(weights, grads)

# mag_rank can be [0, 1, 2]
# taylor_rank can be [1, 0, 2]
```

## What the gate checks

The gate computes the reference scores itself using Python. It checks that both
returned rankings exactly match the oracle rankings for several weight and
gradient tensors.

The gate also verifies that the implementation can distinguish the two
importance measures by checking that the oracle magnitude and Taylor rankings are
different for the tested data. Returning the magnitude ranking for both outputs
will fail.
