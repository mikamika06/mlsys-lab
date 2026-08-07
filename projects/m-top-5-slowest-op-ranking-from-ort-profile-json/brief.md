# Performance Engineering Ticket: ONNX Runtime CPU Profile Analysis Failure

## Symptom
The CPU inference service for our deployed model is experiencing mysterious latency spikes and throughput degradation during production telemetry sessions. To diagnose the bottleneck, the platform team enabled ONNX Runtime profiling (`ort_profile`), but analyzing the generated JSON trace files has proven completely intractable.

First, engineers attempting to find the top performance bottlenecks are overwhelmed by thousands of granular entries, making it impossible to reliably extract a clean top-5 slowest operator ranking without getting bogged down by duplicate node names, sub-node dispatches, or transient initialization overheads.

Second, the reported execution times are distorted because the profiling instrumentation itself introduces significant, unmeasured runtime overhead, skewing latency attribution and causing false alarms about specific operators.

Third, macro-level performance summaries fail to correctly categorize operators into high-level functional groups (such as matrix multiplications, normalizations, elementwise transformations, and reductions), preventing our optimization pipeline from computing accurate time-share percentages and category budgets. As a result, the team is currently flying blind, unable to trust profile reports or prioritize kernel optimizations effectively on CPU inference nodes. We need a robust, automated utility to parse, clean, correct for instrumentation overhead, and classify ONNX Runtime JSON profiles deterministically.
