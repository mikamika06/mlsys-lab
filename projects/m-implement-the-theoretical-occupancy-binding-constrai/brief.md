# Ticket: Kernel Occupancy Optimization and Register Capping Mismatch

## Symptom

During recent performance regression testing of our low-level CUDA kernels on the cluster, profiling logs indicate a recurring discrepancy between our hand-calculated theoretical occupancy values and the actual execution metrics captured by NVIDIA Nsight Compute (`ncu`). Specifically, kernels configured with high register counts are experiencing unexpected performance drops that do not align with our current occupancy models.

Additionally, our automated tuning sweep scripts for selecting the optimal launch register cap—intended to maximize active warps per streaming multiprocessor (SM) without inducing register spilling to local memory—are frequently choosing suboptimal compiler flags. In some cases, the script selects a register limit that forces heavy local memory spills, while in other cases, it leaves considerable occupancy headroom unused because the binding constraint identification logic misclassifies whether registers, shared memory, or block/warp limits are throttling the launch configuration.

Engineers have reported that cross-checking our theoretical occupancy formulas against reference hardware profile reports (`ncu` CSV/JSON logs) fails when trying to isolate the exact limiting resource constraint. We need a robust, deterministic occupancy calculation module that accurately models hardware constraints, correctly identifies the active binding constraint for any given launch configuration, computes the optimal register cap to avoid spilling thresholds, and cleanly validates against our standard set of reference profile reports.
