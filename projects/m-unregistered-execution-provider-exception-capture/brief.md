# Unregistered Execution Provider and IOBinding Overhead in Inference Sessions

## Symptom

When deploying machine learning models using ONNX Runtime CPU and accelerator execution providers, production services intermittently experience abrupt crashes or unhandled runtime errors during session initialization when specific hardware backends are absent from the local runtime binary build. Specifically, attempting to load models configured with hardware-accelerated execution providers on generic worker nodes results in opaque runtime exceptions that bypass standard application-level error handlers, leaving services in an unrecoverable state.

Additionally, performance profiling under high-throughput concurrent inference workloads reveals substantial CPU overhead and memory bandwidth saturation during data transfer phases. Standard execution via the default evaluation API introduces noticeable latency penalties due to repetitive tensor copying and allocation overhead between host memory and runtime buffers, particularly when handling medium-to-large multi-input models. Engineers observe that standard invocation paths fail to leverage pre-allocated device memory effectively, causing unnecessary garbage collection pressure and degraded request latencies compared to optimized binding interfaces.

## Objective

Your task is to implement a robust execution provider exception capture mechanism that gracefully intercepts missing or unregistered provider errors during session creation, and to develop an optimized inference execution path utilizing explicit I/O binding to minimize buffer allocation overhead and data transfer costs. Finally, you must write a comprehensive regression test suite ensuring these invariants are strictly preserved across updates.
