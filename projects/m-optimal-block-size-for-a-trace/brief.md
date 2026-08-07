# Optimal Block Size for a KV-Cache Trace

Production serving logs show unexpected KV-cache memory pressure and latency spikes during peak load. The current PagedAttention system uses a fixed block size across all workloads, leading to high internal fragmentation on short prompts and excessive block-table overhead on long context windows. Furthermore, cache invalidation traces indicate sporadic cache corruptions during sequence extension under severe memory pressure.

Your task is to analyze workload traces, simulate block-level prefix caching, and build triage mechanisms for block table state:

1. **Optimal Block Size Analysis**: Implement trace-level memory footprint and hit-rate analysis to discover the optimal KV block size that minimizes total memory overhead for a given request trace.
2. **Prefix-Cache Simulator**: Build a simulator for LRU prefix caching across block tables, tracking block-level cache hits, evictions, and memory savings under capacity constraints.
3. **Corrupt Block-Table Triage**: Detect and repair corrupted or inconsistent block table mappings (e.g., dangling block references, out-of-bounds indices, and unaligned block boundaries) before they cause silent kernel panics or invalid memory accesses.
4. **Safety Net**: Author comprehensive regression tests in `tests/test_regression.py` that validate trace optimization invariants and catch bugs in block table allocation routines.
