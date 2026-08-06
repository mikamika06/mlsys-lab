# PagedAttention Block-Table Allocation and Internal Fragmentation Analysis

Production inference servers using vLLM's PagedAttention manage Key-Value (KV) cache memory in discrete physical blocks (e.g., 16 or 32 tokens per block) rather than contiguous memory buffers. While dynamic block allocation eliminates external memory fragmentation, it introduces internal fragmentation at the tail end of active sequences.

Your team is optimizing memory footprint and concurrency limits for a vLLM deployment. Currently, capacity estimation models assume linear memory scaling per token without accounting for per-sequence tail block fragmentation. Consequently, sequence concurrency calculations derived from startup memory profiling logs overestimate available capacity under real-world prompt length distributions. Furthermore, the serving engine lacks a deterministic block table allocator capable of tracking allocated physical blocks and mapping dynamic sequence growth.

In this project, you will build a block table allocator for PagedAttention, compute physical memory overhead under internal fragmentation, and parse vLLM startup log signatures to determine maximum concurrent sequence capacity under realistic workloads.

## Deliverables

1. Implement physical block table tracking and calculation functions in `paged_kv/allocator.py` to calculate exact physical blocks needed for variable-length sequence batches while accounting for tail internal fragmentation.
2. Implement a minimal `BlockTableAllocator` class in `paged_kv/allocator.py` that allocates, appends tokens to, and frees physical blocks from a managed pool.
3. Parse vLLM engine startup log signatures and hardware profiles in `paged_kv/startup.py` to derive available KV cache memory and accurately compute max concurrent sequences given sequence length specs and block sizes.
4. Construct a regression test suite in `tests/test_regression.py` that validates allocator state invariants and detects subtle allocation corruptions (such as internal fragmentation under-allocation or improper tail block sharing).
