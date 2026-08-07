## Context

An inference server can only afford to pre-compile a fixed **budget of $K$
distinct sizes** ("buckets") — batch sizes or sequence lengths it
pre-captures (e.g. as CUDA graphs). Every observed request size $s$ that
isn't itself a bucket must be padded up to the smallest chosen bucket
$\ge s$, wasting $(\text{bucket}(s) - s)$ units per request:

$$
\text{total\_waste}(K) = \sum_{s} \text{count}(s)\cdot\big(\text{bucket}(s)-s\big).
$$

For sorted distinct observed sizes $s_1<\dots<s_M$, the optimal $K$-bucket
choice always partitions them into $K$ **contiguous** ranges, each
covered by a bucket equal to the range's largest size (swapping any
non-contiguous choice for the contiguous one never increases waste). That
gives an exact DP: for a range $[i,j]$ covered by bucket $s_j$,

$$
\text{cost}(i,j)=\sum_{t=i}^{j}\text{count}(s_t)\cdot(s_j-s_t),
$$

and the optimal $K$-bucket waste is the minimum, over ways to split
$1,\dots,M$ into $K$ contiguous ranges, of the sum of each range's cost.

**More buckets can only help.** Every valid $K$-bucket partition is also a
valid $(K+1)$-bucket partition (just split one range further), so the
optimal waste is *monotonically non-increasing* in $K$ — but each extra
bucket costs more memory to pre-capture. $K=4$ vs $K=8$ is exactly that
trade-off: does doubling the capture budget meaningfully cut waste on this
workload, or has it already plateaued?

## Task

Implement `compare_k4_k8_waste(sizes, counts)`:

```python
def compare_k4_k8_waste(sizes: list[int], counts: list[int]) -> tuple[int, int, int]:
    ...
```

- `sizes`: 1-D array of distinct observed request sizes (positive ints).
- `counts`: 1-D array, same length, `counts[i]` = observed count for
  `sizes[i]`.

Compute the optimal bucket-selection DP described above **twice** — once
for $K=4$, once for $K=8$ (if there are fewer than $K$ distinct sizes,
using all of them, for zero waste, is optimal for that $K$) — and return

```
(waste_k4, waste_k8, reduction)
```

where `waste_k4`/`waste_k8` are the minimum achievable `total_waste` at
each bucket budget, and `reduction = waste_k4 - waste_k8` (always `>= 0`).

## Example

```python

sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 500]
counts = [10, 10, 10, 10, 10, 10, 10, 10, 10, 1]

compare_k4_k8_waste(sizes, counts)
# waste_k4: 4 buckets must cover 10 distinct sizes -> real waste from
#   padding within each range, plus 500 always needs its own bucket.
# waste_k8: 8 buckets get much closer to one-bucket-per-size -> less waste.
# reduction > 0
```

## What the gate checks

The oracle loads a fixed 20-distinct-size histogram fixture (skewed
counts, mimicking an observed request-size distribution) plus several
seeded synthetic histograms — including one with exactly 8 distinct sizes
(where $K=8$ should reach zero waste while $K=4$ usually still can't) —
and independently computes `(waste_k4, waste_k8, reduction)` via the exact
DP for each, asserting `reduction >= 0` as an internal sanity check on the
oracle itself.

Your returned triple is compared for **exact** equality against the
oracle's on every case; `exact_match` is the fraction of cases that match
and must be `1.0`. Getting either DP wrong (e.g. using non-contiguous
ranges, an off-by-one on which end of a range gets the bucket, or
forgetting to sort the sizes first) will produce a `waste_k4` or
`waste_k8` above the true minimum, which will also throw off `reduction`
even if one of the two individual waste values happens to be right.
