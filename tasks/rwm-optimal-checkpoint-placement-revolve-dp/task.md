## Context

Training large neural networks often requires saving intermediate activations during the forward pass so they can be reused during the backward pass. Checkpointing reduces memory usage, but too few checkpoints causes extra forward computation because discarded activations must be recomputed.

Let $L$ be the number of sequential layers and $C$ be the number of available checkpoints. Define $R(L,C)$ as the minimum number of additional forward-layer evaluations required during the backward pass.

The recurrence is based on splitting the computation at a checkpoint boundary. If the split leaves $k$ layers in the first part, one extra traversal of those $k$ layers is needed before solving the two smaller problems:

$$
R(L,C) = \min_{1 \le k < L} \left(k + R(k,C) + R(L-k,C-1)\right).
$$

The boundary cases are:

$$
R(L,C)=0 \quad \text{when } L \le 1,
$$

and when no checkpoints are available:

$$
R(L,0)=\frac{L(L-1)}{2}.
$$

This dynamic program is the core idea behind optimal checkpoint scheduling methods such as revolve-style algorithms.

## Task

Implement `optimal_recompute(L, C)`:

```python
def optimal_recompute(L: int, C: int) -> int:
    ...
```

Return the minimum number of extra forward-layer evaluations required for a network with $L$ layers and $C$ checkpoints.

Inputs satisfy $L \ge 0$ and $C \ge 0$. The function must return an integer and must compute the optimum, not a heuristic placement.

## Example

```python
print(optimal_recompute(4, 0))
# 6

print(optimal_recompute(4, 4))
# 0
```

For four layers and no checkpoints, the backward pass repeatedly recomputes earlier layers:

$$
3+2+1=6.
$$

## What the gate checks

The gate builds its own dynamic-programming oracle from the recurrence above and compares the returned recomputation count against that oracle on a range of layer and checkpoint values.

A heuristic strategy such as always splitting in half or always using the largest checkpoint segment will fail because it can produce a larger recomputation count than the optimal dynamic program.
