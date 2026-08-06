# Compute the Largest Model-Ctx Pair, Detect Thrash, and Triage Failures on Apple Silicon

Local runner execution on Apple Silicon unified memory architectures frequently encounters unpredictable bottlenecks when pushing hardware limits. Operators running large language models experience severe performance degradation where generation throughput drops drastically mid-stream, accompanied by system-level latency spikes. Additionally, unexpected process terminations occur during high-load inference sessions, leaving ambiguous diagnostic footprints in system logs.

When deploying models near the hardware working-set ceiling, determining the safe upper bound for combined model weight allocation and context window size requires rigorous calculation. Exceeding this boundary without adequate safeguards triggers aggressive operating system paging and swap behavior, resulting in severe performance collapse. Furthermore, distinguishing between low-level GPU allocation errors, unexpected native segmentation faults, and operating system memory reclamation kills is critical for automated recovery systems.

This exercise unit requires you to implement three core subsystems:
1. An automated calculation utility that evaluates model configurations and memory footprints to determine the largest feasible (model, context window) pair under a designated working-set ceiling.
2. A telemetry analysis module that accurately detects swap thrashing and performance degradation from token generation rates and memory pressure indicators.
3. A robust log triage parser that correctly classifies different failure modes—specifically distinguishing Metal allocator failures, native process crashes, and out-of-memory termination events.

You must also author comprehensive regression tests in tests/test_regression.py to ensure your triage logic maintains correct invariants under edge-case conditions.
