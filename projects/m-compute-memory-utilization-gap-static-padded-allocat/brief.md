# Symptom: High Memory Fragmentation and Suboptimal Concurrency in Model Serving

The serving system is experiencing severe memory inefficiency during request handling. Under static sequence allocation, memory must be pre-allocated based on conservative maximum sequence lengths. This leads to massive internal fragmentation, high memory waste, and premature Out-Of-Memory (OOM) errors even when actual token utilization remains low.

Additionally, operators lack an automated, empirical mechanism to determine the true maximum concurrent sequence capacity of local serving instances under dynamic allocations like PagedAttention. Relying on theoretical estimates often fails due to runtime overheads, fragmentation, and KV-cache block allocation mechanics.

To resolve this performance bottleneck, you must build an empirical and analytical toolset that quantifies the compute-memory utilization gap between static padded allocations and PagedAttention KV-cache management, while also providing a simulated OOM-probing framework to discover true max concurrency.

## Requirements

1. **Static vs. Paged KV-Cache Analysis (`gap/utilization.py`)**:
   - Implement `calculate_static_memory()` to compute total bytes allocated when reserving static maximum context lengths for active sequences.
   - Implement `calculate_paged_memory()` to compute actual block-based memory consumption under PagedAttention with fixed block sizes.
   - Implement `compute_utilization_gap()` to measure wasted memory bytes, effective utilization ratios, and the compute-memory efficiency gap.

2. **Empirical OOM Probing Simulation (`gap/probing.py`)**:
   - Implement `probe_max_concurrency()` to empirically find the maximum number of concurrent sequences a vLLM-style local instance can support before running OOM given total available KV-cache memory, block size, sequence length distributions, and safety margins.

3. **Regression & Safeguard Testing (`tests/test_regression.py`)**:
   - Write tests that validate the calculation of utilization gaps and ensure the system correctly flags under-utilization caused by static padding overestimation.
