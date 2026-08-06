# Ticket: Unexplained TTFT Spikes and Worker Imbalance in Disaggregated Serving

**From**: Operations & Performance Team
**Context**: Long-context disaggregated serving (Llama-3 32k prompt workload)

We recently migrated our long-context serving infrastructure from monolithic nodes to disaggregated prefill-and-decode clusters. While benchmark scripts predicted higher throughput, initial telemetry under realistic 32k prompt workloads shows severe TTFT (Time-To-First-Token) degradation and uneven resource utilization.

Prefill workers (P-nodes) frequently stall after finishing model execution, unable to dispatch requests because network links are saturated. Meanwhile, Decode workers (D-nodes) sit idle waiting for KV cache buffers to arrive over the interconnect. Our initial static P:D replica ratio calculations assumed instantaneous KV transfer, ignoring interconnect bandwidth bottlenecks and fixed protocol overheads.

We need a dedicated analysis and simulation module to:
1. Quantify exact KV cache transfer time versus prefill compute time across different prompt lengths (specifically 32k context) and interconnect bandwidths.
2. Calculate optimal P:D replica ratios that account for transfer latency alongside compute time.
3. Simulate request throughput, TTFT, TPOT, and node utilization in a disaggregated serving setup with connector overhead.
