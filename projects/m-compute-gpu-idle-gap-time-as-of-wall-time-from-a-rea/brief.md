We are debugging a low-level LLM inference engine where performance profiling reveals unexpected stalls and micro-stabs in GPU execution time. During high-throughput generation phases, our monitoring systems flag that the primary GPU stream is dropping utilization and showing intermittent idling periods.

When we export the profiling data via Nvidia Nsight Systems (nsys), the timeline reveals gaps between consecutive kernel launches that do not align with expected host-device synchronization latencies. Furthermore, during steady-state request processing, certain execution traces display a distinct sawtooth pattern in kernel execution lengths, which points towards recurrent, hidden synchronization points across streams or unnecessary explicit synchronizations.

Your task is to implement a robust low-level profiling parser that ingests raw timeline records and trace logs. Specifically, you must:
1. Compute the exact GPU idle-gap time as a percentage of total wall time from a real Nsight Systems capture log, accounting for overlapping kernel executions and stream concurrency.
2. Provide a warm-up module that computes GPU idle-gap time and overall utilization percentage from a synthetic stream of kernel lists with varying durations and launch intervals.
3. Count the exact per-iteration synchronization points by analyzing the structural properties and frequency of the sawtooth patterns in a real nsys timeline trace, ensuring our regression test suite catches any accidental reintroduction of redundant sync points.
