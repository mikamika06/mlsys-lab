# BRGeMM Microkernel Dispatch and Roofline Efficiency Bottlenecks

During micro-benchmarking of CPU transformer GEMM layers on modern vector hardware, performance monitoring reported severe throughput variance across tensor shapes. Under certain tile configurations, measured performance unexpectedly drops well below predicted hardware peak compute throughput. Conversely, for other matrix shapes, runtime performance appears memory-bandwidth bound even when memory bandwidth ought to be underutilized.

Deep trace analysis indicates two underlying software flaws in the low-level execution stack:

1. The batch dispatch runtime fails to properly reconstruct the low-level BRGeMM (Batch-Reduce GEMM) microkernel API call sequence. Specifically, when tiling problem dimensions $(M, N, K)$ into tile blocks $(m_b, n_b, k_b)$, the batch reduction counts and pointer offsets for $A$ and $B$ matrices are computed incorrectly for partial boundary tiles and multi-tile loops.
2. The analytical performance tool miscalculates memory traffic for cache-blocked matrix multiplication. By failing to account for matrix tile reuse factors across outer tile loops, the memory traffic estimation yields invalid arithmetic intensity figures, skewing the theoretical roofline ceiling calculations and reporting incorrect hardware efficiency percentages.

We need a clean, correct implementation of the BRGeMM microkernel dispatch sequence generator in `brgemm/dispatch.py` and the DRAM memory traffic and roofline ceiling analyzer in `brgemm/roofline.py`. Finally, a suite of regression tests in `tests/test_regression.py` must be authored to verify dispatch invariants and detect memory traffic calculation faults.
