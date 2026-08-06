# Incident Report: Distributed All-Reduce & DDP Bucket Mechanics Failure

## Ticket Summary
During recent scale-up evaluations of our distributed training pipeline using PyTorch's Distributed Data Parallel (DDP) framework, we observed severe synchronization anomalies, incorrect gradient all-reduce outputs, and suboptimal communication scheduling across multi-node setups. Specifically, ring all-reduce communication primitives fail to handle wrap-around pointer arithmetic correctly when the ring size is non-power-of-two or when buffer chunks are misaligned, leading to silent data corruption or deadlocks. Furthermore, our performance tuning runs indicate that the default `bucket_cap_mb` parameter causes suboptimal overlapping of backward pass computation with gradient communication. Finally, DDP's default reverse-order bucket assignment mechanism is misbehaving during gradient accumulation and bucket registration, causing gradients to be dispatched out of sequence and violating the strict dependency requirements of our gradient synchronization schedule.

## Symptoms & Observations
* **Ring All-Reduce Corruption:** Scatter-reduce and all-gather phases in the ring communication loop exhibit off-by-one errors in rank indexing and send/recv buffer offsets, causing corrupted tensor values across ranks.
* **Bucket Capacity Bottlenecks:** Naive configurations of `bucket_cap_mb` lead to massive communication stalls or excessive small-message overhead during the backward pass. We need an empirical sweep mechanism to evaluate communication throughput across varying bucket sizes.
* **Reverse-Order Bucket Misalignment:** Reconstructing DDP's reverse-order bucket assignment reveals that parameters are currently being registered in forward order instead of reverse order of module definition, destroying the crucial optimization where communication overlaps with the tail of the backward pass.

## Action Required
1. Fix the ring all-reduce implementation to ensure robust chunk rotation, correct index wraparound, and exact numeric matching against the golden reference all-reduce algorithm.
2. Implement a rigorous bucket capacity sweep utility (`bucket_cap_mb`) that evaluates real DDP communication and computation overlap across multiple candidate sizes.
3. Reconstruct and verify DDP's reverse-order bucket assignment logic, ensuring that parameters are bucketed correctly in reverse order and that your test suite rigorously catches any regressions where this ordering invariant is violated.
