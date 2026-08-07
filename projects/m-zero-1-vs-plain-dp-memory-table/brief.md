Ticket #3819: Memory spikes and severe workload imbalance during distributed LLM training runs.

Our training jobs on 8-GPU nodes are suffering from severe GPU memory pressure and unpredicted Out-Of-Memory (OOM) errors during the optimizer step, even when using ZeRO Stage 1.
In our telemetry dashboard, we noticed two major symptoms across runs:

First, our cluster provisioning calculator is producing wildly incorrect estimates for optimizer memory consumption under ZeRO Stage 1 compared to standard Data Parallelism (DP). Operators cannot reliably predict how much memory ZeRO-1 frees up when scaling world sizes or changing optimizer precision formats (Adam vs FP32 master weights).

Second, during active training, parameter optimizer updates are severely imbalanced across ranks. Rank 0 regularly encounters memory spikes that exceed its VRAM budget while other ranks remain underutilized. Inspecting rank parameter slices revealed that flattened parameter buffer offsets are misaligned or overlapping incorrectly across rank boundaries, and unpartitioned layer tensors assigned to processes are piling up on single GPUs due to poor rank assignment logic.

We need a unified memory estimation module and corrected parameter partitioning routines for our distributed optimizer runtime. The current helper functions in `zerodp/` need to be implemented and verified against exact reference calculations and regression tests.
