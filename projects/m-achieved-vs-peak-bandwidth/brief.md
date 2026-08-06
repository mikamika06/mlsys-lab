# Ticket: Inconsistent Bandwidth Utilization Metrics in Tiled Attention Kernels

## Symptom
During recent roofline profiling of custom tiled attention kernels (`rw2-flashattention` block execution) on our datacenter accelerators, the telemetry dashboard reported invalid performance metrics. Specifically, under certain batch and sequence configurations, reported memory bandwidth utilization exceeded 100% of theoretical peak hardware DRAM bandwidth (e.g., 2.4 TB/s on a 1.5 TB/s peak memory bus). Conversely, for small block row sizes ($B_r$) with high SRAM recomputation counts, the reported memory bandwidth efficiency dropped to near zero despite high execution latency.

The automated roofline analysis harness rejected the profiling reports due to inconsistent byte transfer accounting between naive HBM materialization and tiled recomputed accesses.

## Expected Deliverable
Implement precise memory transfer and bandwidth efficiency calculation functions in `bandwidth/tracker.py` and roofline analysis utilities in `bandwidth/roofline.py`. Ensure total DRAM byte counts account for both naive intermediate matrix materialization ($S, P$) and tiled recomputed loads across tile iterations ($T_r = \lceil N / B_r \rceil$). Finally, write regression tests in `tests/test_regression.py` to ensure metric validation fails when memory byte transfer equations under-count recomputation passes.
