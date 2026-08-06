# Edge Runner Performance and Portability Anomaly Report

## Ticket Summary
Field engineering teams report inconsistent throughput and deployment failures when running GGUF-quantized models across heterogeneous edge hardware nodes.

When benchmarking identical GGUF model files on the same physical host, performance telemetry indicates substantial generation throughput variances between llamafile executable runs and Ollama daemon deployments. On developer workstations, llamafile delivers higher generation tokens-per-second, while on staging instances Ollama exhibits distinct IPC overhead and context processing times. The engineering team currently lacks automated attribution tooling to determine whether these performance deltas stem from microarchitecture feature flag dispatch (such as AVX-512 vs AVX2), HTTP/IPC IPC payload serialization, or thread pool configuration differences.

Furthermore, deployment teams are unable to reliably select and validate local runner packaging for target air-gapped environments. In several remote field deployments, single-file llamafile binaries compiled for x86_64 fail to launch on target Linux nodes without descriptive error output, whereas they run without issue on identical host hardware running slightly different kernel or page size configurations.

We need a benchmarking attribution engine, a deployment runner selection module, and a binary portability diagnostic utility to standardize runner evaluation and debug host compatibility issues.
