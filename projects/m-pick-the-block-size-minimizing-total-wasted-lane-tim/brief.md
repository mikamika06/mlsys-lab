# Pick the Block Size Minimizing Wasted Lane-Time

In high-performance kernel design—such as when tuning block sizes for vector additions or tiled kernels—choosing an inappropriate hardware block size can lead to massive overhead due to inactive lanes and tail padding. When processing an array of size `N` across GPU warps or threads, selecting a candidate `BLOCK_SIZE` introduces two distinct sources of waste:

1. **Masked Predication Overhead (Tail Waste):** The remaining elements `N % BLOCK_SIZE` do not completely fill the final block. The unaligned boundary forces active execution lanes to operate under masks, wasting clock cycles on inactive threads.
2. **Launch & Dispatch Overhead (Fixed Block Penalty):** Launching too many tiny blocks incurs high threadblock scheduling overhead, while launching overly large blocks wastes register resources and inflates execution latencies per block launch.

Your task is to analyze workload dimension configurations and compute the optimal `BLOCK_SIZE` from a set of candidate choices that minimizes total wasted lane time.

## Objective

1. Compute the total wasted lane-time metric for given tensor lengths $N$ and candidate block sizes.
2. Build an efficient search algorithm that identifies the optimal block size index (`argmin_index`) across various workload shapes.
3. Write a regression suite that verifies bounds checking, edge cases ($N$ perfectly divisible by `BLOCK_SIZE`), and optimal selection invariants.
