# Ticket: Unpredictable llama-bench throughput degradation under context expansion and layer offloading

## Symptom
Users deploying LLM workloads with `llama.cpp` and running standard `llama-bench` suites report severe, non-linear text generation throughput drops as active sequence context increases. Benchmarks executed with context depth `-d 512` achieve high generation throughput, but scaling to `-d 8192` causes performance to degrade far more aggressively than team hardware estimates predicted.

In addition, platform engineers are unable to accurately forecast generation throughput when tuning GPU offloading parameters (`-ngl`). Offloading partial model weights across host CPU RAM and GPU VRAM produces erratic scaling behavior. In several instances, partial offloading configuration (`-ngl 16` out of 32 layers) resulted in bottlenecked generation rates that performed worse than expected simple interpolation between zero offloading (`-ngl 0`) and full offloading (`-ngl 32`).

## Requirements
We need an analytical modeling library inside our tooling package `llamaperf` to evaluate generation throughput scaling without requiring hardware-bound execution loops:
1. `llamaperf.decay.measure_context_decay` must compute per-token Key-Value (KV) cache memory footprint, predicted generation throughput across varying context depths `-d`, and throughput decay ratios relative to baseline context depth.
2. `llamaperf.offload.compare_offload_throughput` must calculate memory bandwidth-limited token throughput for two offload layer configurations (`-ngl1` vs `-ngl2`), accounting for host CPU memory bandwidth, GPU memory bandwidth, and activation transfer overhead across the PCIe bus during partial offloading.
3. Regression tests in `tests/test_regression.py` must validate that context depth expansion strictly reduces generation throughput and that offloading more layers to higher-bandwidth GPU VRAM improves throughput.
