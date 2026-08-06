Our local chat server is thrashing its KV cache. Whenever a user edits a previous message or the system drops an older message to fit within the context limit, the server recomputes the entire KV cache from scratch. The logs show `Cache hit: 0 tokens` even when 90% of the prompt is identical to the previous turn!

We should be able to reuse the prefix of the prompt that hasn't changed. In a block-based KV cache (like PagedAttention), we need to predict exactly which tokens—and subsequently, which cache blocks—survive the context shift.

You need to implement two parts:
1. `caching/prefix.py`: Find the longest common prefix of tokens between a new prompt and a set of cached prompts.
2. `caching/blocks.py`: Translate this prefix matching to actual cache blocks. A block is fully reusable only if ALL its tokens perfectly match the corresponding positions in the new prompt. The reusable blocks must form a strictly contiguous prefix from the start of the sequence.

Finally, write a robust test in `tests/test_regression.py` that verifies blocks are strictly matched in order, and a mismatch strictly truncates any further reuse. This prevents disjoint blocks from being falsely claimed as a continuous prefix.
