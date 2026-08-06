# Production Outages from OOM Crashes and Latency Spikes in Parallel Serving

Our 8-GPU vLLM serving clusters are experiencing random Out-Of-Memory (OOM) crashes and tail latency spikes on multi-layer Mixture-of-Experts (MoE) workloads. Cluster telemetry shows that static parallelism layouts are either overflowing GPU memory headroom during peak concurrency bursts or assigning suboptimal Tensor Parallelism (TP) / Pipeline Parallelism (PP) / Data Parallelism (DP) combinations that suffer high inter-node communication delays. Furthermore, severe expert-routing load imbalance in MoE layers is creating straggler GPUs that bottleneck pipeline execution.

To restore service stability and throughput, we need an automated layout chooser and a routing imbalance metric tool.

## Requirements

1. **Memory Split Checking**: Implement a memory checker in `layout/chooser.py` that verifies whether a candidate TP/PP split fits within GPU VRAM limits. The memory model must account for per-GPU model weight distribution, KV cache allocation across TP and PP partitions, and activation headroom.
2. **EP Straggler Quantification**: Implement an Expert Parallelism (EP) straggler calculation in `layout/imbalance.py` to evaluate load imbalance from token-to-expert routing histograms.
3. **Layout Selection**: Build `select_layout` in `layout/chooser.py` to find the layout index (`argmin_index`) over valid 8-GPU TP/PP/DP candidate combinations that yields the lowest predicted latency while guaranteeing memory feasibility.
4. **Regression Safeguards**: Write automated tests in `tests/test_regression.py` validating that memory limits are enforced during layout selection and that expert imbalance calculations remain accurate.
