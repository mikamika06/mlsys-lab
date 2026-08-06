# Ticket: CoreML Conversion Pipeline Latency and Compatibility Tuning on Apple Silicon

## Symptom
While deploying a custom edge-optimized vision and sequence model converted via `coremltools` into our target macOS/iOS environment, our performance and deployment pipeline is encountering two major bottlenecks:

1. **Unstable Latency and Device Utilization**: In certain test scenarios, inference requests running through the CoreML runtime exhibit unexpectedly high latencies and fail to leverage the unified memory architecture efficiently. Specifically, switching runtime execution options between CPU-only execution and full hardware acceleration (`compute_units=CPU_ONLY` versus `compute_units=ALL`) yields unpredictable execution profiles, making it difficult to guarantee real-time frame rates on edge hardware.
2. **Layout Opacity and Deployment Target Failures**: When compiling models for older or constrained deployment environments with low minimum deployment targets (e.g., legacy iOS/macOS versions), the model compilation phase intermittently fails due to unsupported operator versions or unhandled op-availability errors. Furthermore, engineers currently lack an automated way to inspect the internal `.mlpackage` directory layout to verify weight serialization, metadata structure, and compiled asset manifests against the official CoreML specification before deployment.

## Goal
We need to establish a rigorous, programmatic pipeline in our coremltools integration package that:
- Accurately compares prediction latency and execution behavior profiles under different compute unit configurations (`compute_units=CPU_ONLY` versus `compute_units=ALL`) using controlled mock inference inputs and invariant checks.
- Inspects and validates the internal directory layout and structural components of an `.mlpackage` bundle against the documented CoreML specification.
- Reproduces, diagnoses, and safely resolves op-availability errors caused by low minimum deployment targets by adjusting target flags or mapping fallback paths.
- Implements a robust regression test suite that catches regressions in model packaging layout and deployment target compatibility constraints.
