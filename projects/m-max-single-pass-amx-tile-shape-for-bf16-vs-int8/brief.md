# Ticket: AMX Tile Configuration and Throughput Mismatch in Low-Level CPU Kernel

## Symptom

During micro-benchmark profiling of low-level CPU inference kernels on modern Intel Xeon architectures (Sapphire Rapids and newer), matrix multiplication routines targeting the Advanced Matrix Extensions (AMX) coprocessor are failing to achieve expected hardware saturation. Specifically, when dispatching matrix block loads and tile configurations for bfloat16 (bf16) and 8-bit integer (int8) workloads, the kernel encounters execution stalls, illegal configuration faults, or severe throughput degradation compared to theoretical peak performance.

When executing dense GEMM primitives, the tile loader attempts to configure tile register dimensions that exceed hardware capacity or misalign with the 64-byte row stride limitation of the tile configuration palette. Furthermore, comparative benchmarks against AVX-512 FMA fallback paths report unexpected performance anomalies where the hardware matrix multiplication unit (TMUL) either underperforms vector units due to excessive tiling overhead or fails classification checks for single-pass execution eligibility.

Engineers attempting to tune micro-kernels report that matrix block shapes $(M, N, K)$ that appear valid under standard vector register assumptions violate AMX structural constraints, leading to incorrect tile configuration register states (`TILECFG`), silent execution throttling, or fallback to scalar/vector paths. The system lacks a robust module to compute exact single-pass tile boundaries, evaluate TMUL arithmetic intensity and peak operational throughput relative to AVX-512 FMA baselines, and classify matrix block shapes for single-pass AMX tileability.
