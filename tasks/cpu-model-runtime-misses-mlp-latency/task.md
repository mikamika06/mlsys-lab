## Context

A modern out-of-order processor can have multiple cache misses **in flight**
simultaneously.  This capability is called **memory-level parallelism** (MLP).
When MLP independent miss streams overlap, the effective latency per miss drops.

A simple analytical model for a memory-bound kernel is:

$$
T_{\text{model}} \;=\; \frac{N_{\text{misses}}}{\text{MLP}} \;\times\; t_{\text{miss}}
$$

where $N_{\text{misses}}$ is the total number of unique cache-line misses produced
by the access trace, MLP is the number of outstanding misses the hardware can
service in parallel, and $t_{\text{miss}}$ is the latency of a single DRAM
access in cycles.

Consider a pointer-chasing traversal of $n$ nodes, each of `node_size` bytes,
laid out sequentially in memory starting at byte address $0$.  Node $i$ resides
at byte address $i \times \text{node\_size}$.  The access order is:

$$
0,\;\text{node\_size},\;2 \times \text{node\_size},\;\ldots,\;(n-1) \times \text{node\_size}
$$

The cache is fully-associative-within-set (set-associative), parameterised by
`line_bytes` (cache-line size in bytes), `sets` (number of sets), and `ways`
(associativity).  Two addresses that map to the same cache line and the same
set compete for the same slot; an access that cannot be satisfied from the
current tags is a **miss**.

## Task

Implement:

```python
def modeled_runtime(n_nodes, node_size, line_bytes, sets, ways, mlp, miss_latency):
    """Return the modeled execution time in cycles (float)."""
```

Your function must:

1. Generate the sequential pointer-chase byte-address trace for `n_nodes` nodes
   of `node_size` bytes each (addresses $0, \text{node\_size}, \ldots$).
2. Determine the number of cache misses that trace produces in a set-associative
   cache with the given `line_bytes`, `sets`, and `ways`.
3. Return the modeled runtime $\frac{N_{\text{misses}}}{\text{MLP}} \times t_{\text{miss}}$.

You may use `arena.cachesim.simulate` to obtain the miss count, or derive it
analytically.  Either approach is acceptable as long as the final value is
correct.

## Example

```python
# 256 nodes, 128 bytes each -> every node is on its own cache line
# 64-byte lines, 16 sets, 4 ways -> 256 misses
# MLP=4, miss_latency=200 -> (256/4)*200 = 12800.0
modeled_runtime(256, 128, 64, 16, 4, 4, 200)  # -> 12800.0
```

## What the gate checks

The grader runs `arena.cachesim.simulate` on the same trace for **five**
different parameter sets, computes the reference runtime for each, and measures
the maximum relative error:

$$
\text{rel\_err} = \max_k \frac{\lvert T_{\text{learner}}^{(k)} - T_{\text{ref}}^{(k)} \rvert}{T_{\text{ref}}^{(k)}}
$$

The gate passes when $\text{rel\_err} \le 0.01$ (at most 1 % error on every
configuration).
