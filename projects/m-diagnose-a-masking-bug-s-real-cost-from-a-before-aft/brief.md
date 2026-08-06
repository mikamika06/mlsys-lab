We are running a Triton-based FlashAttention kernel on an NVIDIA H100 GPU and looking at Nsight Compute (ncu) profiling reports comparing a clean run against one with an accidental masking bug enabled.

In the clean kernel, attention scores are computed, masked efficiently via block-level bounds, and written out. In the broken run, an extra explicit conditional masking branch or redundant memory predicate instruction is introduced inside the inner loop for every single element, even when entirely within bounds.

Your task is to parse both Nsight Compute report summaries (provided as structured CSV/JSON metrics generated from ncu profiles in `ref.py`), compute the exact performance metrics—such as active warp occupancy, specific pipeline stall reasons, instruction mix deltas (like increased predicate instructions or branch instructions), and the resulting DRAM/SRAM throughput impact—and write an analysis tool that isolates the real cost of this masking bug.

You need to implement three milestones:
1. Parse and extract key metrics from Nsight Compute sections (SpeedOfLight, WarpStateStats, ComputeWorkloadAnalysis) comparing before and after files.
2. Quantify the exact cost breakdown: compute the precise instruction overhead, divergence penalty, and pipeline stall cycle increases attributed to the masking bug.
3. Write a regression test suite that catches regressions in profiling analysis correctness, ensuring any faulty metrics computation or misclassification of stall reasons is flagged.
