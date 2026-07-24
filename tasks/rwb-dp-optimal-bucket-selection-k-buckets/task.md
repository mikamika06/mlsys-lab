## Context

An inference server can only afford to pre-compile (CUDA-graph capture, or
similarly pre-allocate) a fixed **budget of `K` distinct sizes**
("buckets") — batch sizes, sequence lengths, whatever the runtime shape
is. Every request of size $s$ that isn't itself a bucket has to be padded
up to the *smallest chosen bucket that is $\ge s$*, wasting
$(\text{bucket} - s)$ units of compute. Given the observed distribution
of request sizes (a histogram of `size -> count`), which `K` sizes should
become buckets to minimize total padding waste?

$$
\text{total\_waste} = \sum_{s} \text{count}(s) \cdot \big(\text{bucket}(s) - s\big), \qquad
\text{bucket}(s) = \min\{\, b \in \text{buckets} : b \ge s \,\}
$$

Every request must have *some* bucket to round up to, so the largest
observed size must itself be a bucket. Because sizes round **up** to the
nearest bucket, the optimal bucket set — for sorted distinct sizes
$s_1 < s_2 < \dots < s_M$ — always partitions the sizes into $K$
**contiguous** ranges, each covered by one bucket equal to its range's
largest size (an exchange argument: swapping a bucket for the next size
up only ever reduces waste for everything it covers, and reordering a
non-contiguous choice never wastes less than the contiguous one). That
turns it into a classic DP: for a range $[i, j]$ covered by bucket
$s_j$, its cost is
$\text{cost}(i,j) = \sum_{t=i}^{j} \text{count}(s_t)\cdot(s_j - s_t)$,
and the optimal $K$-bucket cost is the minimum over ways to split
$1 \ldots M$ into $K$ contiguous ranges of the sum of each range's cost.

## Task

Implement `select_buckets`:

```python
def select_buckets(size_histogram: dict[int, int], k: int) -> tuple[list[int], int]:
    ...
```

- `size_histogram`: `{size: count}`, positive ints, one entry per
  distinct observed size.
- `k`: the bucket budget (positive int). If `k` is at least the number of
  distinct sizes `M`, using all `M` sizes as buckets (zero waste) is
  optimal — return all of them.

Return `(buckets, total_waste)`:

- `buckets`: the chosen bucket sizes, each one drawn from
  `size_histogram`'s keys, achieving the **minimum possible**
  `total_waste` as defined above (any optimal choice is accepted — ties
  are graded on the achieved waste, not on which exact set you pick).
- `total_waste`: the actual total padding waste your `buckets` produce,
  by the formula above.

## Example

```python
size_histogram = {1: 5, 2: 5, 3: 1, 100: 1}
select_buckets(size_histogram, k=2)
# Best 2-bucket split: {3, 100} -- covering {1,2,3} with bucket 3 costs
# 5*(3-1) + 5*(3-2) + 1*0 = 15, and {100} alone costs 0.
# A split like {2, 100} would cost 5*(2-1) + 1*(100-3) = 5 + 97 = 102 for
# covering {1,2} with bucket 2 and {3} rounding all the way up to 100 --
# worse, because it fails to keep the contiguous "cheap" range together.
```

## What the gate checks

The grader builds several `(size_histogram, k)` scenarios — hand-picked
skewed histograms (one huge outlier size, tight clusters, `k` larger
than the number of distinct sizes) and several from a seeded NumPy
generator — and computes the true minimum `total_waste` independently
with the DP over sorted distinct sizes described above, never calling
your function.

For each scenario, the grader **never trusts your reported `total_waste`
at face value**: it recomputes the waste your own `buckets` actually
produce (rounding every observed size up to the nearest bucket you
returned) and requires that to match what you reported, that every
bucket you return is one of the histogram's own sizes, and that it
covers the maximum observed size. `exact_match` is the fraction of
scenarios where all of that holds **and** your buckets' actual waste
equals the DP's true minimum; the gate requires `1.0`. A greedy or
uniform-spacing bucket choice, an off-by-one in which end of a range
gets the bucket, or forgetting that the largest size must be covered
will all land above the true minimum on at least one scenario.
