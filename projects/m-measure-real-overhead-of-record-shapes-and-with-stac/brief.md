Our training and inference evaluation scripts running on Apple Silicon (Mac) instances are experiencing unexplained throughput degradation during profiling runs. Benchmark reports show severe overhead spikes when developers enable PyTorch profiler options such as `record_shapes` and `with_stack`. In some cases, profiling runs take over twice as long as raw execution, but the exact overhead contribution of each flag—and how they compound when enabled simultaneously—remains unquantified across our workload suites.

Furthermore, several automated profiling traces recorded across 10-step evaluation runs exhibit wall-clock durations that suggest profiling was active across the entire workload rather than strictly during the scheduled active steps. We suspect that certain training loop wrappers fail to invoke `prof.step()`, preventing the profiler state machine from transitioning out of the `RECORD` or `ACTIVE` states and inadvertently imposing full-run overhead.

You need to implement an overhead measurement and trace diagnostic toolkit:
1. Parse benchmark trace samples to compute the exact per-event overhead (in nanoseconds) contributed by `record_shapes` and `with_stack` independently and when combined.
2. Formulate the mathematical relationship that derives the per-event overhead ratio relative to unprofiled baseline event time at which total profiled execution wall-clock time doubles (`throughput_ratio`).
3. Analyze execution traces to detect missing `prof.step()` / `schedule()` invocations that leave full-run profiling active.
4. Provide a regression test suite verifying that trace analysis correctly flags traces missing scheduled step transitions.
