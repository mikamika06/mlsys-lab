# Aggregated vs Disaggregated Serving under a Fixed GPU Budget

Our production cluster serves high-throughput LLM workloads on a fixed pool of homogeneous GPUs. Under high prompt-token ratios, our current unified (aggregated) deployment experiences severe time-to-first-token (TTFT) degradation and high inter-token latency (ITL) variance when prefill and decode phases run on the same workers.

To evaluate whether disaggregating prefill and decode workers improves serving performance under identical resource budgets, we need an analytical and event-driven simulator. The simulator must compare an aggregated pool of $N$ GPUs against a disaggregated pool split into $P$ prefill GPUs and $D$ decode GPUs (where $P + D = N$).

You need to build a simulation module that models request arrival, prefill chunk execution, KV-cache transfers across the inter-device network, and decode iteration batching.

## Symptoms & Target Behavior

- **TTFT Spikes:** Under heavy decode load, prefill requests are stalled or forced to compete with existing decode batches, degrading first-token latency.
- **Underutilized Network vs Compute:** Allocating too many nodes to prefill leaves decode workers starved, whereas too few prefill nodes creates a bottleneck on incoming requests.
- **Metric Gates:** Your simulator must correctly compute latency ratios (disaggregated TTFT / aggregated TTFT, and disaggregated ITL / aggregated ITL) to identify the optimal $(P, D)$ split for a target workload budget.

## Tasks

1. Implement the queueing and scheduling dynamics for both aggregated serving (shared prefill/decode on all $N$ GPUs) and disaggregated serving ($P$ prefill nodes, $D$ decode nodes with cross-node KV cache transfer overhead).
2. Measure total latency, TTFT, ITL, and compute the disaggregated-to-aggregated latency ratio across fixed GPU budgets.
3. Write a regression test suite in `tests/test_regression.py` that verifies system behavior under varying network bandwidths and detects incorrect KV transfer latency calculations.
