# Cache Blocking and Register Tiling Derivation

After deploying a custom low-level matrix multiplication kernel to our CPU inference pipeline, performance monitoring reveals significant performance degradation on large matrices ($N \ge 2048$). While small matrices achieve reasonable FLOPS, throughput drops precipitously as problem dimensions grow, accompanied by an unusually high L2 cache miss rate and severe memory bandwidth saturation. Additionally, hardware counter profiling shows high instruction counts per MAC operation, suggesting inefficient register usage in the inner execution loop.

Your task is to analyze the cache memory hierarchy and vector register bank parameters to construct an optimal execution blocking schedule for dense matrix multiplication ($C = A \cdot B$).

You need to implement parameter derivation routines that calculate optimal tile dimensions based on microarchitectural constraints. Specifically:
1. Reconstruct register-tile dimensions $(m_R, n_R)$ from total vector register count and SIMD vector width to maximize register reuse and minimize load/store instructions.
2. Calculate cache-blocking parameters ($k_C$ and $m_C$) targeting L2 cache capacity, ensuring that active working panels for $A$ and $B$ fit alongside output accumulators without cache thrashing or premature evictions.
3. Quantify performance throughput across matrix dimensions, contrasting naive triple-loop execution with cache-blocked execution to verify scaling efficiency.
4. Provide a regression suite in `tests/test_regression.py` that validates parameter boundaries and memory footprint constraints against improper blocking configurations.
