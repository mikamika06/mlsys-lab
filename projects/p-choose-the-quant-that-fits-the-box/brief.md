# Ticket: Select Quantization Recipes for 16GB, 36GB, and 64GB Target Hardware

We are planning a rollout of our core generative model across a heterogeneous fleet of target machines consisting of 16 GB, 36 GB, and 64 GB memory capacities. Management has requested a rigorous, data-driven prescription for which quantization format and bit-width configuration should be deployed on each hardware tier.

Our current baseline is an unquantized FP16 deployment, which requires excessive headroom and limits concurrent request batching. We need to evaluate the precise trade-offs between memory footprint, inference throughput, and quality degradation (measured via perplexity and Kullback-Leibler divergence against the FP16 baseline) for various bit-width choices, including 2, 3, 4, 5, 6, and 8 bits per weight (bpw).

The output must be strictly quantitative and deterministic. We require an automated pipeline that computes exact file sizes, measures actual peak memory consumption under realistic workloads, quantifies perplexity penalties, benchmarks execution speed across configurations, outputs a definitive threshold-based selection table, and provides an auto-selection script for new hardware nodes. No speculative opinions or heuristic estimates will be accepted; every recommendation must be backed by empirical harness runs.
