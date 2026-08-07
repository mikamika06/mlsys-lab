# Distributed Training Overlap Analysis & Saturation Profiling

We observed significant variance in step times and poor scale-out efficiency on our distributed training cluster. Engineers report that gradients are slow to synchronize, but raw trace summaries from PyTorch Profiler / NCCL logs show conflicting numbers for communication overhead. Some nodes appear heavily communication-bound, while others appear to overlap all collective communication almost perfectly with backward compute passes.

Without a grounded baseline, the team cannot determine whether performance drops stem from bad bucket size configurations, un-overlapped tail collectives, or inaccurate event labeling in raw traces.

To solve this, we need a unified profiling tool that:
1. Reconstructs execution timelines directly from raw execution events, accurately resolving compute vs. communication overlap without double-counting concurrent stream execution.
2. Computes the **Theoretical Minimum Step Time** (the absolute lower bound imposed by critical-path compute and network communication) and the **Overlap-Efficiency Score** to measure how well collective communication is hidden behind backward compute.
3. Analyzes communication bucket-size scaling to pinpoint the **Empirical Bucket-Size Overlap Saturation Point**—the optimal bucket size where communication latency and pipeline overlap reach maximum effective efficiency.

Build the trace reconstruction, overlap score calculator, bucket saturation analyzer, and regression safety tests to validate distributed profiling invariant checks.
