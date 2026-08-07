# Investigating unexpected VRAM spikes and compile-time trade-offs in our JAX serving loop

Operators report that our high-concurrency JAX state-update pipeline suffers from severe VRAM inflation during peak traffic hours, frequently triggering unexpected Out-Of-Memory (OOM) exceptions. Initial telemetry suggests that buffers passed into state-update functions are not being reclaimed or invalidated as expected under JIT compilation, leading to duplicate allocations coexisting in device memory. Furthermore, the team lacks a systematic method to evaluate whether the initial compilation latency of our compiled update kernels is actually amortized by the request volume, or if we are losing overall serving efficiency compared to running purely under eager execution.

We need to instrument our memory lifecycle checks, analyze peak memory savings under donation semantics, and implement precise request break-even computations for ahead-of-time and just-in-time compiled serving loops.

Specifically, you must:
1. Implement utility logic to verify whether a donated JAX buffer is correctly invalidated post-execution.
2. Calculate peak memory consumption and savings ratios when applying `donate_argnums` on state-update operations.
3. Determine the exact request count break-even point where compiled execution overhead is completely offset by per-step latency gains over cumulative eager execution.
