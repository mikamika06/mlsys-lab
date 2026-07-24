## Context

A triangular solve finds a vector $x$ from a triangular system

$$
Lx=b ,
$$

where $L \in \mathbb{R}^{N \times N}$ is lower triangular. Forward substitution computes each
entry using previously solved values:

$$
x_i = \frac{b_i - \sum_{j=0}^{i-1} L_{ij}x_j}{L_{ii}} .
$$

For row $i$, the summation has $i$ multiply-add contributions. Counting one multiplication and
one addition as one multiply-add pair, the number of multiply-add operations is

$$
\sum_{i=0}^{N-1} i = \frac{N(N-1)}{2}.
$$

The same count applies to backward substitution with an upper triangular matrix. This task
models the arithmetic work rather than executing the solve.

## Task

Implement `triangular_solve_flops(n)`:

```python
def triangular_solve_flops(n: int) -> int:
    ...
```

Return the number of multiply-add pairs required for a triangular solve of size $n$.

The input $n$ is a non-negative integer. The function must return an integer count.

## Example

```python
triangular_solve_flops(0)
# 0

triangular_solve_flops(4)
# 6
```

## What the gate checks

The gate builds the expected count by simulating the row-wise triangular solve operation
structure and compares it with the returned value. The result must exactly match the oracle
count for several matrix sizes.
