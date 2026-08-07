# Ticket: Radix Tree Prefix Caching Engine Instability under Heavy Concurrent Loads

## Symptom
During high-throughput multi-tenant serving sessions utilizing radix tree-based prefix caching and tree-of-thought sampling workflows, our inference cluster exhibits severe memory degradation and unexpected cache invalidations. Specifically, long-running prefix nodes are occasionally evicted prematurely while active child references still point to them, resulting in dangling pointers, segmentation faults in the underlying custom block allocator, and incorrect prefix reuse counts.

Furthermore, operators report that the reported memory overhead metrics for storing large numbers of distinct sequence prefixes do not scale linearly or predictably with node counts, frequently underestimating actual heap consumption by significant margins. Finally, performance counters tracking token savings from fork reuse during tree-of-thought branch expansion show inflated efficiency metrics that contradict downstream generation throughput, suggesting that prefix matching logic fails to properly account for shared child divergence points.

## Requirements
1. Identify and fix the reference implementation flaw in the refcount-safe node eviction routine where parent-child reference updates and zero-ref cleanup are ordered incorrectly.
2. Implement precise tree memory overhead calculations that correctly model node metadata structures, pointer arrays, and hash map bucket allocations against varying numbers of stored sequences.
3. Compute exact token savings from fork reuse during multi-branch tree-of-thought sampling scenarios without double-counting shared prefixes.
4. Provide a robust regression test suite that catches any regressions where refcount safety or window bounds are violated.
