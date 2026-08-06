# Ticket: Non-Deterministic Gradients and Memory Overhead in Attention Backward Pass

## Observed Symptoms
During multi-node training runs using custom attention kernels with dropout enabled ($p > 0$), downstream gradient checks intermittently report severe numerical mismatches across accumulation steps, even when input tensors and global random seeds remain identical. Distributed worker nodes computing identical mini-batches diverge in their query, key, and value gradients ($dQ$, $dK$, $dV$).

Additionally, our execution profiler reports unexpected peak memory consumption spikes scaling quadratically with sequence length ($O(N^2)$) during the backward pass when using high sequence length configurations. Memory usage logs suggest that intermediate attention score matrices are unexpectedly persisting in memory rather than being recomputed on demand during backward backpropagation.

Finally, the execution planner lacks accurate operational accounting metrics for the exact arithmetic floating-point operations (FLOPs) incurred when enabling activation recomputation in attention backward passes. This prevents the scheduler from accurately balancing memory savings against kernel recomputation overhead.

## Requirements
1. Implement a manual attention backward pass `attention_backward` and deterministic `generate_dropout_mask` function in `attnbwd/backward.py` that computes exact analytical gradients $dQ, dK, dV$ given $Q, K, V, dO$, dropout probability $p$, and random seed.
2. Implement backward FLOP counting and recomputation overhead metrics in `attnbwd/overhead.py`.
3. Implement regression tests in `tests/test_regression.py` that verify dropout mask reproducibility and verify that non-deterministic dropout mask generation is reliably caught.
