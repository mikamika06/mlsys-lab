# Symptom: Inconsistent Batch Processing and SPMD Numerical Divergence

The distributed compilation team reported numerical discrepancies and performance anomalies when migrating per-example model evaluation routines to vectorized batch transformations (`vmap`) and SPMD device reductions (`pmap` + `psum`). Downstream pipeline validation failed because vectorized batch execution produced unexpected output tensors when scaled across varying batch sizes, while single-device reference execution remained consistent.

Additionally, data-parallel synchronization across simulated multi-device topologies showed silent state divergence. When running gradient aggregation steps across simulated 4-CPU device groups, individual workers appear to retain un-reduced local state instead of computing globally synchronized sums, causing loss curves to drift after several batch steps.

We need a verified batching harness that checks vectorized outputs against an explicit per-example loop reference, benchmarks vectorization speedup across batch size scaling factors, and executes a multi-device SPMD all-reduce (`pmap` + `psum`) to ensure numerical consistency across 4 simulated CPU devices.
