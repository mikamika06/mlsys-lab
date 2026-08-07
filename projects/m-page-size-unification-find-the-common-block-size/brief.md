# Page-Size Unification: Find the Common Block Size

When serving hybrid transformer architectures with heterogeneous layer configurations (varying head dimensions, key-value head counts, or precision types), allocating KV cache pages with independent block sizes per layer leads to severe memory fragmentation and complex allocator design.

To maintain a uniform physical memory page allocator across all execution units, the serving framework must unify physical page layouts across layers. A unified page layout selects a common block size in tokens $B$ such that the resulting block memory size in bytes for every layer satisfies low-level physical alignment constraints (e.g., SIMD or cache line alignment requirements, such as 64-byte boundaries).

Your task is to implement page-size unification utilities that determine the optimal unified block size across layer configurations, compute page allocations for active sequences, and calculate memory waste.

## Symptoms / Requirements
- Different layer types (e.g., full attention vs. GQA/MHA layers with distinct $H_{kv}$ or $D_{head}$) produce non-aligned byte footprints when partitioned into token blocks.
- Unaligned block byte sizes trigger inefficient unaligned memory access or allocation failures in aligned pool managers.
- You must find the candidate token block size $B$ that satisfies byte alignment for all layer definitions in a model configuration.
- You must calculate sequence memory allocations and measure internal fragmentation (wasted byte capacity due to block padding and sequence tail remainder).
- You must provide regression tests in `tests/test_regression.py` that catch incorrect block size selection algorithms that fail alignment constraints.
