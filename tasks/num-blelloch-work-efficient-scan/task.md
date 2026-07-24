## Context

A prefix scan transforms a sequence into cumulative partial results. For an input
sequence $x$ of length $n$, an exclusive prefix sum produces

$$
y_i = \sum_{j=0}^{i-1} x_j ,
$$

with $y_0 = 0$. The final element of the output is the sum of all input values
except the last one.

The Blelloch scan algorithm computes this result using an up-sweep phase and a
down-sweep phase. The up-sweep builds a reduction tree by combining partial
sums. The down-sweep propagates prefix values back through the tree while
swapping and adding child values.

For an array with length $n = 2^k$, the algorithm performs $O(n)$ additions,
which is work-efficient compared with repeatedly computing every prefix sum.

## Task

Implement `blelloch_scan(values)`:

```python
def blelloch_scan(values: list[int]) -> list[int]:
    ...
```

The input length is always a power of two. Return a new list containing the
exclusive prefix sums computed by the Blelloch algorithm. Do not modify the
input list.

The implementation should perform the tree-style up-sweep and down-sweep
rather than calling built-in cumulative sum helpers.

## Example

```python
values = [3, 1, 7, 0, 4, 1, 6, 3]
result = blelloch_scan(values)

# [0, 3, 4, 11, 11, 15, 16, 22]
```

## What the gate checks

The gate compares the returned list against a reference Blelloch scan computed
inside the checker. The comparison is exact because all fixtures use integer
values. The implementation must return the same exclusive prefix scan for
multiple input sizes and must not rely on modifying the provided input.
