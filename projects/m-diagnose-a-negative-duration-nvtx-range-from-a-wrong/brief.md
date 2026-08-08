# Symptom Ticket: Profile Timeline Distortions and Phase Self-Time Miscalculations

We are observing severe anomalies in our low-level model profiling pipelines across two distinct profiling environments.

First, during CUDA runtime analysis on our multi-threaded training workers, Nsight Systems trace exports report NVTX ranges with negative durations—specifically showing end timestamps occurring prior to start timestamps. On several iterations, `nvtxRangePop` invocations appear to close ranges pushed on different threads or pop from empty thread-local stacks, corrupting the execution timeline visualization and yielding garbled span metrics.

Second, on our macOS developer workstations running PyTorch `record_function` profile exports, engineers attempting to isolate bottlenecks find that phase execution reports merely aggregate total wall-clock durations. Because nested annotations (such as `layernorm` inside `attention` inside `forward`) overlap in time, the reported phase timings double-count child operations, concealing which execution phase actually dominates computational self time.

We need a dedicated utility in `nvtxprof` to parse raw NVTX trace logs, maintain proper thread-local stacks to diagnose wrong-thread pops and negative-duration ranges, and analyze Mac Chrome-format trace exports to compute accurate phase self times and rank top execution phases.
