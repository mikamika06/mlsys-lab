# Edge Model Profiling and ANE Execution Diagnostics

Our mobile vision deployment team is observing unexpected latency spikes when serving a multi-head detector model on Apple platforms. While synthetic benchmarks suggested optimal runtime under unified execution configs, real-device profiling reveals severe frame drops during batch inference on target hardware.

Initial telemetry points to inconsistent execution unit dispatch across different `MLComputeUnits` configurations. Rather than scaling linearly with hardware accelerator availability, the total pipeline execution time degrades significantly under certain execution unit combinations. Diagnostics indicate that specific graph operations fail to target the Apple Neural Engine (ANE), forcing the Core ML runtime to fall back to the CPU and incur costly memory transfers between distinct execution domains.

We need a structured diagnostic suite to systematically evaluate unit latency across all four compute configurations, isolate the specific graph node triggering CPU fallback segments, and construct a predictive classifier that identifies ANE-ineligible operations prior to graph compilation.
