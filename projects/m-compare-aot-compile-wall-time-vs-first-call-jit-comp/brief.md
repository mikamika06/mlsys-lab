# Ticket: Unpredictable Warm-up Latency and Unmonitored Compiler Optimization Shifts

## Symptom

Production serving clusters are experiencing unpredictable latency spikes during cold startup and model deployment transitions. Engineers report that initial inference requests are taking significantly longer than expected, but existing telemetry cannot distinguish between explicit ahead-of-time (AOT) staged compilation overheads, JIT tracing delays, and actual kernel execution duration. When model updates are deployed with modified compiler flag options, warm-up times fluctuate wildly without any clear indication of which compilation stage is responsible.

Additionally, performance optimization efforts are hampered by a lack of visibility into intermediate representation changes. When tuning XLA compiler flags or altering precision settings, engineers cannot easily inspect or quantify how the generated StableHLO operation counts change across flag sets. Unexpected operation expansion (such as un-fused elementwise operations or missing matrix multiplication lowering rules) passes unnoticed until performance degrades in production.

Without automated profiling utilities to isolate AOT vs. JIT compilation timings and diff StableHLO operation structures, cluster operators cannot establish reliable latency budgets or verify optimization pass effectiveness before rolling out new model artifacts.
