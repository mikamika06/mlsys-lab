An internal benchmarking report for the text-generation serving infrastructure shows significant discrepancies between nominal hardware costs and actual cost-per-token metrics under diurnal traffic patterns. Engineering leadership requires an automated cost and capacity calculator to validate production deployments on vLLM and select optimal instance configurations.

When evaluating current vLLM nodes, naive estimations based on peak throughput significantly underestimate the actual cost per million tokens when accounting for realistic load fluctuations, target headroom, and strict p99 latency Service Level Objectives (SLOs). Furthermore, manual configuration selection across varying Tensor Parallelism (TP) levels, quantization schemes, and GPU instance types has resulted in over-provisioned clusters.

You are tasked with building a modular cost and capacity planning module in `capacity/`. The module must:
1. Calculate the real-world cost per million output tokens derived from empirical throughput measurements and hourly instance pricing.
2. Determine exact GPU cluster replica counts required to sustain diurnal traffic profiles while maintaining a mandatory 30% capacity headroom safety margin.
3. Evaluate benchmark candidate matrices across 12 distinct configurations (combining GPU types, quantization models, and TP degrees) to select the most cost-effective hardware strategy meeting a target p99 latency SLO.
4. Provide unit tests in `tests/test_regression.py` that enforce capacity safety invariants against under-provisioned planning logic.
