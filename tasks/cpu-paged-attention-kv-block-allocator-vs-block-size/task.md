## Context

Paged attention implementations for large language models maintain *key-value (KV) caches* whose entries are blocks of hidden-state vectors. Each decoded token reserves
a few bytes in GPU or CPU memory; to avoid frequent reallocations, these entries are managed by a simple *paged allocator*. Allocation granularity — the **block size** —
controls internal fragmentation (wasted gaps within blocks) and also table overhead (page table entries, meta‑data).

If each token uses $s$ bytes and you round allocations to the nearest multiple of block size $b$, fragmentation leads to $\frac{b - s \bmod b}{b}$ unused fraction per allocation on average, while smaller $b$ increases the page table’s overhead.

We want to choose a block size that minimizes wasted memory, i.e. maximizes

$$
\text{size\_ratio} = \frac{\text{useful bytes}}{\text{allocated bytes}}
$$

for the modeled workload of token sizes.

## Task

Implement:

```python
def choose_kv_block_size(token_sizes: list[int], table_overhead_per_block: int) -> tuple[int, float]:
    """
    Choose an integer block size (power of two) that maximizes useful / allocated
    bytes for the given workloads.
    Returns (best_block_size, achieved_ratio).
    """
```

For each candidate block size $b \in \{16, 32, 64, 128, 256, 512, 1024\}$ bytes:

* Each token of size `s` consumes `ceil(s / b) * b` total bytes.
* The total allocated bytes = sum over tokens + (#blocks) * `table_overhead_per_block`,
  where #blocks is the total number of `ceil(s / b)` across all tokens.
* The useful bytes = sum of the `token_sizes`.

Return the block size giving highest ratio `useful / allocated`. In ties, pick the smaller block size.

## Example

```python
sizes = [80, 96, 200, 500]
best_b, ratio = choose_kv_block_size(sizes, table_overhead_per_block=16)
# best_b might be 128, ratio ≈ 0.91
```

## What the gate checks

The grader recomputes ratios for all candidates using its own formula
(never a hard‑coded table) and compares the student's result with the reference’s best
choice. The metric is `size_ratio` = (ratio_student / ratio_ref). Solutions within
10% of the optimal (≥ 0.9) pass. Deterministic and hardware‑independent.
