## Context

In many data-processing pipelines, individual samples (or batches) are assigned to **buckets** of fixed capacities $b_0 \le b_1 \le \dots \le b_{k-1}$.
A sample of size $s$ must be placed into the smallest bucket whose capacity is at least $s$; if no bucket is large enough, it goes into the largest bucket $b_{k-1}$ (the *eager* fallback).

A common implementation mistake is to assign the sample to the bucket whose capacity is *nearest* to $s$ (minimising $|b_i - s|$). When the nearest bucket is smaller than $s$, the sample cannot fit, which may silently truncate or drop data.

## Task

The function `bucket_assign(sizes, buckets)` takes a list of sample sizes and a sorted list of bucket capacities and returns a list of bucket indices (0‑based) using the correct **ceiling** assignment rule:

- For each size $s$, find the smallest index $i$ such that $b_i \ge s$.
- If all buckets are smaller than $s$, use the index of the largest bucket $k-1$.

The provided starter code contains a buggy implementation that uses *nearest* assignment. Fix it.

```python
def bucket_assign(sizes: list[int], buckets: list[int]) -> list[int]:
    ...
```

- `sizes`: list of non‑negative integers.
- `buckets`: list of positive integers, sorted in increasing order. At least one bucket.

Returns a list of integers, one per size, each in $[0, k-1]$.

## Example

```python
sizes = [5, 7, 12, 30]
buckets = [8, 12, 20]

# Correct assignment:
#   5  -> bucket 0 (8)
#   7  -> bucket 0 (8)
#  12  -> bucket 1 (12)    (smallest bucket that is >= 12)
#  30  -> bucket 2 (20)    (no bucket >= 30, use the largest)
assert bucket_assign(sizes, buckets) == [0, 0, 1, 2]
```

## What the gate checks

The gate metric `exact_match` equals $1.0$ only when the returned indices exactly match those of a reference implementation on a set of representative test cases (including samples that are larger than the largest bucket, samples exactly on bucket boundaries, and random sequences). The nearest‑neighbour bug fails on at least one of these cases, producing a grade of $0.0$.
