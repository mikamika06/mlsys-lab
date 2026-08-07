# Ticket: Investigate Pipeline Parallelism Degradation and Stage Memory Bottlenecks

We are observing significant divergence between our theoretical cluster utilization and the actual delivered throughput when running large transformer models using pipeline parallelism (PP) with 1F1B and interleaved 1F1B schedules.

During our recent training runs across multiple nodes, operators have reported two major symptoms: first, the overall cluster throughput calculations do not match the expected analytical formulas established in the Megatron-LM PTD-P literature, leading to incorrect sizing decisions. Second, specific training jobs are experiencing unexpected out-of-memory (OOM) errors on isolated pipeline ranks even when the global memory footprint appears balanced across the job allocation. Analysis of raw execution logs suggests that certain pipeline stages bear a disproportionate activation memory burden due to uneven parameter or microbatch assignments, but we lack a systematic way to pinpoint the exact imbalanced stage from log outputs.

Additionally, as we evaluate transitioning from standard 1F1B schedules to interleaved 1F1B schedules to reduce the pipeline bubble, we need a rigorous verification framework to confirm that the peak activation memory at a matched bubble fraction conforms to expected theoretical limits, and to ensure our regression testing correctly traps any broken memory or throughput assumptions.

Please implement the required modules under `pipelp/` and complete the regression tests under `tests/test_regression.py` to diagnose these discrepancies, reproduce the correct throughput and bubble formulas, identify imbalanced stages from logs, and safeguard our pipeline memory invariants.
