# Diagnostic: ktransformers Expert Placement & Offloading Performance

We have received reports from field deployments of ktransformers where inference throughput degrades unexpectedly on heterogeneous CPU-GPU setups using Mixture-of-Experts (MoE) models. Production teams suspect that the expert placement strategy is misaligning CPU and GPU execution, leading to excessive transfer overhead and poor hit rates on cached expert weights.

To resolve these issues, you need to analyze recorded schedule logs, compute cache efficiency over expert selection traces, and model offloading trade-offs.

## System Symptoms
- In recorded schedule logs, expert placement decisions cannot be traced back reliably to GPU memory budgets, causing unexpected falling back to CPU execution.
- System metrics show high CPU-GPU bus contention during token generation, suggesting low LRU expert cache reuse on hot experts.
- Benchmark telemetry comparing offload strategies (`offload-all` vs `offload-first-N-layers`) shows inconsistent latency scaling as sequence lengths grow.

## Objectives
1. **Reconstruct Expert Placement**: Implement `reconstruct_placement` to reconstruct the exact GPU vs CPU expert assignment map per layer given an available VRAM budget, expert weight sizes, and expert assignment logs.
2. **LRU Cache Hit Rate**: Implement `simulate_lru_cache` to calculate the exact cache hit rate on a real MoE expert selection trace under fixed expert cache capacity.
3. **Offloading Latency Strategy**: Implement `evaluate_offload_latency` to compute estimated token processing latency across offload-all versus offload-first-N-layers strategies on llama.cpp-style benchmarks.
4. **Regression Safeguard**: Author a test suite in `tests/test_regression.py` that verifies expert placement and cache invariance, ensuring offloading logic catches invalid placement configurations.
