# AMX Tile Configuration and Throughput Modeling

During recent benchmark profiling of low-level CPU GEMM kernels utilizing Intel AMX (Advanced Matrix Extensions), several operational anomalies were observed across different execution phases. First, tile configuration initialization routines (`LDTILECFG`) occasionally result in silent memory corruption or general protection faults depending on how the 64-byte palette configuration structure is populated in memory. Second, despite high theoretical TMUL (Tile Matrix Multiply) compute ceilings, actual microbenchmarks show substantial deviation from theoretical peak TFLOPS/TOPS when vectorizing matrix operations across AMX and AVX-512 execution paths. Finally, roofline analysis indicates unexpected stall cycles during matrix multiplication loops, suggesting an imbalance between tile load/store overheads and active TMUL compute execution.

To fix and characterize these performance bottlenecks, you need to implement a structured tile configuration builder, derive an analytical AMX throughput model comparing AMX-Tile to AVX-512 execution, and construct a detailed micro-architectural time-share breakdown for TMUL compute versus memory operations.

Your objective is to implement:
1. `amx/config.py`: Correct binary encoding of the 64-byte `tilecfg` structure used by `LDTILECFG` for Palette 1, including header setup (palette ID, start_row) and per-tile dimension descriptors (`bytes_per_row` and `rows`).
2. `amx/model.py`: An analytical performance model that calculates theoretical peak throughput for AMX vs. AVX-512 based on system hardware parameters, estimates measured speedups, and calculates the compute-to-memory operational time share during tile matrix operations.
3. `tests/test_regression.py`: A test suite validating structural and operational invariants of your tile configuration generator and throughput modeling functions.
