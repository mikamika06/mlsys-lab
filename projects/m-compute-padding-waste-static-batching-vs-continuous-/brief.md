# Compute Padding Waste: Static Batching vs Continuous Batching

## Symptom

Our LLM serving cluster is experiencing low GPU tensor core utilization and high latency variance under mixed-length workload traffic. Metrics indicate a large gap between theoretical TFLOPs and effective compute spent on useful token generation.

Preliminary analysis points to static rectangular batching: short sequences in a batch are padded with zeros to match the longest sequence, causing the forward pass to perform matrix multiplications over non-functional padding tokens. Meanwhile, requests arriving while a batch is executing are queued, inflating time-to-first-token (TTFT) and inter-token latency (ITL).

We need to quantify this padding waste, simulate serving throughput and latency comparing static batching against continuous batching (iteration-level scheduling), and compute the real dollar cost efficiency (throughput-per-GPU-dollar) across recorded production serving logs.

## What You Need To Do

Implement the core scheduling simulator and analysis tools under `batchsim/`:

1. **`batchsim/padding.py`**:
   - Implement `calculate_padding_waste(batches: list[listFILE: brief.md
```markdown
# Debugging Low Throughput and High Latency in Model Serving Batchers

## Symptom
Your production LLM inference service is running with severe performance degradation. Monitoring telemetry indicates high latency spikes and unacceptably low token throughput per dollar, despite high GPU utilization metrics. The infrastructure team suspects that static padding overhead and suboptimal scheduling in the serving pipeline are wasting compute capacity on uninformative padding tokens.

To fix this, you must analyze and replace the naive static batching mechanism with dynamic continuous batching (iteration-level scheduling). Furthermore, you need to build auditing tools that ingest serving logs, simulate both batching strategies head-to-head on real workload traces, and compute exact throughput-per-GPU-dollar metrics.

## Task
1. In `batching/waste.py`, implement `compute_padding_waste()` to analyze request batches (prefill prompt lengths and decode completion lengths) under static versus continuous batching, calculating padded token overhead and wasted FLOPS ratios.
2. In `batching/simulator.py`, implement `simulate_batcher()` to execute discrete-event simulations for both static and continuous scheduling modes. Compute per-request end-to-end latency, time-to-first-token (TTFT), inter-token latency (ITL), and total system throughput.
3. In `batching/cost.py`, implement `compute_cost_efficiency()` to parse serving log records and derive normalized token throughput per GPU-dollar across static and continuous deployment configurations.
4. In `tests/test_regression.py`, implement regression tests that validate your batching waste calculations and verify that continuous batching consistently achieves higher token throughput and lower padding waste than static batching across heterogeneous sequence length distributions.
