# Ticket: Unstable Distributed Training Runs with ZeRO Offloading Stalls and Memory OOMs

During large-scale model training using ZeRO offloading, several training clusters have reported severe training throughput degradation and intermittent Out-of-Memory (OOM) failures.

When attempting to scale context length or batch size, GPU VRAM fills up unpredictably, forcing lower layer weights and optimizer states into CPU host RAM and NVMe storage. However, tensors appear to be assigned to offload storage tiers indiscriminately without accounting for per-tensor access frequencies, transfer latencies, or physical storage limits on host memory and NVMe arrays. Consequently, high-frequency active parameters are placed on slow NVMe devices, triggering massive execution stalls during backward passes.

Furthermore, CPU Adam parameter update benchmarks show high execution latency on multi-core host nodes, and ZeRO-Infinity offload logs indicate low prefetch hit rates and severe NVMe queue congestion.

We need a unified module to calculate optimal per-tensor target device placements (GPU vs CPU vs NVMe) based on capacity and bandwidth cost models, model CPU Adam throughput speedups, and parse NVMe log traces to diagnose prefetch performance and stall bottlenecks.
