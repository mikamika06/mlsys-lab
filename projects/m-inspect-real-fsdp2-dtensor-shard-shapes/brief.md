# FSDP2 DTensor Sharding and Resharding Profile Discrepancies

## Problem
During our initial migration from PyTorch FSDP1 to PyTorch 2 FSDP2 per-parameter sharding, distributed model initializations are producing unexpected tensor slice offsets and silent GPU memory spikes across worker ranks.

Specifically, when inspect tools analyze DTensor placements across our 8-GPU worker nodes, non-divisible dimension-0 parameter shapes (e.g. sequence lengths or hidden dimensions not evenly divisible by world size) fail to align with downstream buffer assumptions. Node operators report that memory usage post-forward pass remains unexpectedly elevated under non-default reshard settings, causing out-of-memory (OOM) errors during the backward pass on sequence-parallel runs.

## Objectives
1. Implement DTensor shard metadata inspection (`fsdpshards/sharding.py`) to accurately compute rank local shard shapes, global offsets, and element counts for uneven mesh sharding.
2. Quantify FSDP1 padded sharding versus FSDP2 uneven unpadded dim-0 chunking (`fsdpshards/padding.py`) to track exact memory overheads and wasted padding bytes.
3. Profile persistent and peak parameter memory allocations under `reshard_after_forward=True` vs `reshard_after_forward=False` (`fsdpshards/memory.py`).
4. Provide regression tests in `tests/test_regression.py` that catch memory accounting bugs and non-contiguous rank slice coverage.
