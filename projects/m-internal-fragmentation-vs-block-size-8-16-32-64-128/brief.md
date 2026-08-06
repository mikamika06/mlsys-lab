Our new block allocator for the KV cache serving engine is exhibiting high memory usage in production. We suspect two separate issues: structural waste (internal fragmentation due to block sizes) and outright memory leaks (blocks that are allocated but never returned to the pool).

I've dumped a trace of the allocator events for a small workload. We need you to build a parser that finds the leaked blocks. We also need to quantify the internal fragmentation given a batch of sequence lengths and candidate block sizes (8, 16, 32, 64, 128), so we can tune the allocator.

Finally, you need to implement the last piece of our serving path: flattening the physical slot mappings for a ragged batch. This mapping translates every token's logical position in its sequence into a physical slot in the KV cache so our attention kernels know exactly where to read and write.
