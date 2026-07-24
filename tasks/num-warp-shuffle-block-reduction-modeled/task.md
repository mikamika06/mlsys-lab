## Context

A CUDA block reduction combines values owned by many threads into one result. A naive reduction repeatedly writes intermediate values to shared memory. A shuffle-based reduction keeps values inside warp registers and exchanges values with warp shuffle operations.

For a warp of size $W$, a tree reduction uses offsets

$$
\frac{W}{2}, \frac{W}{4}, \dots, 1 .
$$

At each step, a lane adds a value received from another lane. The number of reduction stages is

$$
\log_2(W).
$$

In this modeled task, a block has $B$ threads and the input contains exactly $B$ values. The implementation returns both the reduction result and a modeled memory-operation count. The count represents the expected CUDA behavior:

- each thread performs one global read,
- each warp writes one partial sum to shared memory,
- if multiple warps exist, the first warp reads the partial sums,
- warp shuffle exchanges are counted as register-level accesses.

The goal is not to launch CUDA code, but to implement the algorithmic model of a shuffle-based block reduction.

## Task

Implement `block_reduce(values, block_size, warp_size=4)`:

```python
def block_reduce(values: list[float], block_size: int, warp_size: int = 4) -> tuple[float, int]:
    ...
```

The input list length is always equal to `block_size`. The function must return:

1. the sum of all values as a `float`,
2. the modeled access count as an `int`.

Use a tree-style warp reduction model. Do not use a full shared-memory reduction where every reduction stage writes and reads intermediate values.

## Example

```python
values = [1.0, 2.0, 3.0, 4.0]
result, accesses = block_reduce(values, 4, 4)

# result == 10.0
# accesses models:
# - four global reads
# - one warp shuffle tree reduction
```

## What the gate checks

The gate computes the expected numeric result using a NumPy sum oracle and computes the expected modeled access count from the shuffle reduction algorithm.

The `exact_match` gate requires the returned reduction result and modeled count to match the oracle behavior.

The `modeled_access_count` gate requires the implementation to use the shuffle-based access model rather than a naive shared-memory reduction model. A naive implementation with extra shared-memory traffic will produce a different count.
