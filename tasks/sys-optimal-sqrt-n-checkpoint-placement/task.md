## Context

Activation checkpointing trades memory for recomputation. A chain of $n$ layers can store selected intermediate activations and recompute missing activations during the backward pass.

For this task, a checkpoint placement is modeled by the set of saved layer boundaries. If there are $k$ checkpoints, the peak saved activation memory is proportional to $k$. The recomputation cost is modeled by the longest interval without a checkpoint.

Let the checkpoint positions be a sorted list

$$0 < c_1 < c_2 < \dots < c_k < n.$$

The chain is split into segments with lengths

$$c_1,\ c_2-c_1,\ \dots,\ c_k-c_{k-1},\ n-c_k.$$

The modeled memory access cost is

$$C = k + \max(\text{segment lengths}).$$

Balancing the number of checkpoints against recomputation gives a minimum near the square-root tradeoff because increasing $k$ lowers the maximum segment length while increasing memory usage.

## Task

Implement `optimal_checkpoints(n)`:

```python
def optimal_checkpoints(n: int) -> list[int]:
    ...
```

Return a sorted list of checkpoint layer boundaries for a chain with $n$ layers.

Requirements:

- Every returned checkpoint must satisfy $1 \le c < n$.
- The list must be strictly increasing.
- The returned placement should minimize the modeled cost
  $$C = k + \max(\text{segment lengths})$$
  where $k$ is the number of checkpoints.
- Use integer arithmetic only.

The grader evaluates the returned placement against an independently computed optimum. It does not require a unique placement when several placements have the same cost.

## Example

```python
n = 10
checkpoints = optimal_checkpoints(n)

# One valid answer:
# [3, 6, 8]
#
# Segments:
# 3, 3, 2, 2
#
# Cost:
# 3 + max(3, 3, 2, 2) = 6
```

## What the gate checks

The gate computes the optimal modeled cost by searching the possible checkpoint counts and constructs a reference placement algorithmically.

The returned placement is scored by

$$\frac{C_{\mathrm{candidate}}}{C_{\mathrm{optimal}}}.$$

The `modeled_mem_access` metric must be at most $1.05$, so a placement within $5\%$ of the oracle optimum passes.
