# Tensor Parallel Feasibility and Inter-GPU Traffic Diagnostics

Our production LLM serving cluster has been experiencing silent deployment failures and unexpected interconnect bottlenecks during scale-out. When deploying new fine-tuned model architectures across variable GPU tensor parallel (TP) degree configurations, cluster nodes occasionally crash during initialization or suffer severe throughput degradation far below theoretical compute capacity.

Initial operational logs indicate two distinct root causes:
1. Model initialization crashes when attempting to shard attention heads or intermediate feed-forward layers across TP ranks where the dimensions are not evenly divisible.
2. High inference latency on multi-GPU setups due to unoptimized collective communication traffic (All-Reduce volume per token and aggregated per second) and pipeline execution bubbles (PP) in hybrid TP+PP deployments.

We need a unified validation and diagnostic utility to preemptively assess deployment feasibility and quantify communication overhead.

Your task:
- Implement TP validation routines in `tpval/feasibility.py` to check whether attention head counts, key-value head counts, and intermediate hidden sizes can be evenly sharded across target TP degrees.
- Implement communication and execution profiling routines in `tpval/traffic.py` to compute All-Reduce byte volume per token, network bandwidth requirements at target throughputs, and Pipeline Parallelism (PP) bubble fractions.
- Write a regression test suite in `tests/test_regression.py` that validates TP feasibility checks and traffic calculation invariants, ensuring incorrect shardings are caught and invalid PP schedules are flagged.
