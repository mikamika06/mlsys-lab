## Context

In a CUDA-capable GPU, threads execute in groups of 32 called *warps*.  
A kernel launch is defined by a grid dimension and a block (thread block) dimension. Each thread within a block has an index `threadIdx.x` ranging from `0` to `blockDim.x-1`. If the number of threads in a block is not a multiple of 32, the last warp will contain idle lanes that do nothing for the remaining cycles. These idle lanes are often referred to as *wasted lanes*.

The wasted‑lane fraction for a single block is

$$
\text{fraction} \;=\;\frac{32\,\lceil B/32\rceil - B}{32\,\lceil B/32\rceil},
$$

where $B$ is the number of threads in the block.  
When several blocks are launched, each block suffers the same waste independently, so the overall wasted‑lane fraction is identical to that of one block.

## Task

Implement a function `wasted_lane_fraction` that, given a grid dimension and a thread block dimension, returns:

1. The number of warps per block (rounded up).
2. The total number of warps launched across all blocks.
3. The wasted‑lane fraction for the entire kernel rounded to six decimal places.

```python
def wasted_lane_fraction(grid_dim: int, block_dim: int) -> tuple[int, int, float]:
    ...
```

Your implementation should not rely on any external libraries beyond Python’s standard library.

## Example

```python
>>> wasted_lane_fraction(2, 50)
(2, 4, 0.218750)
```

Explanation:

* `warps_per_block = ceil(50/32) = 2`
* `total_warps = 2 * 2 = 4`
* `wasted_lanes = 64 - 50 = 14`
* `fraction = 14 / 64 ≈ 0.218750`

## What the gate checks

The grader evaluates your function on several hard‑coded test cases.  
It compares the returned tuple exactly (including rounding of the fraction).  
Any deviation or runtime error results in a failure.
