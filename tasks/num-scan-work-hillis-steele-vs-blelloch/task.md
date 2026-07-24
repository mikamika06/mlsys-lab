## Context

A prefix scan combines a sequence of values using an associative operation such as
addition. For an input of length $N$, an inclusive sum scan produces

$$
y_i = \sum_{j=0}^{i} x_j .
$$

Parallel scan algorithms trade off the number of additions performed against the
available parallelism.

The Hillis-Steele scan performs a sequence of distance-doubling steps. At step
$k$, values are combined with elements $2^k$ positions away. The modeled number
of additions is the number of active pairs across all steps:

$$
\sum_{k=0}^{\lceil \log_2 N \rceil - 1} \max(0, N - 2^k).
$$

The Blelloch scan uses an up-sweep and down-sweep tree. For a power-of-two
working size $M$, it performs one addition for each internal tree edge in each
phase:

$$
2(M-1).
$$

This task models the amount of add work, not wall-clock runtime.

## Task

Implement `scan_work(n, algorithm)`:

```python
def scan_work(n: int, algorithm: str) -> int:
    ...
```

Return the modeled number of addition operations for a parallel prefix scan of
length $n$.

The `algorithm` argument is either `"hillis_steele"` or `"blelloch"`.

For Hillis-Steele, count additions by iterating through the distance-doubling
passes. A pass with distance $d$ performs one addition for every index
$i \ge d$.

For Blelloch, model the tree size as the smallest power of two $M$ such that
$M \ge n$. Count additions from both the up-sweep and down-sweep phases.

Raise `ValueError` for unsupported algorithm names or non-positive sizes.

## Example

```python
scan_work(8, "hillis_steele")
# 17

scan_work(8, "blelloch")
# 14
```

## What the gate checks

The gate builds the expected count from an independent reference implementation
of the two scan work models and compares it exactly. The returned metric
`modeled_access_count` must be $1.0$.
