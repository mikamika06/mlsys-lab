When generating text using our custom LLM engine, we are seeing degraded output quality that gets worse as the sequence length grows. The models use a paged KV cache (similar to vLLM) to manage memory efficiently during decoding. We strongly suspect that the attention mechanism is occasionally pulling in random noise or failing to correctly align with the sequence context lengths when reading from the block tables.

To isolate the issue, we want to rigorously prove that our paged attention implementation is mathematically equivalent to standard, contiguous attention.

First, we need to build a utility that reconstructs the contiguous KV cache from the block tables, and implement a standard un-paged attention function to serve as our baseline.
Second, we'll write the paged attention function that operates natively on the block tables, and ensure its output matches the contiguous baseline with near-zero error.
Finally, we will write a regression test that checks if the paged attention correctly respects the actual `context_lens`. The test must fail if the implementation inadvertently includes garbage padding data from the final partially-filled block, which is a common source of subtle numerical bugs in paged attention.
