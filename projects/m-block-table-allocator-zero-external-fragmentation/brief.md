The local test serving engine for our new MoE model is running out of memory much faster than we mathematically calculated. We have a 16GB limit, and given our typical context lengths, we should be able to hold around 50 concurrent requests. Instead, the engine starts throwing OOM exceptions at roughly 20 concurrent requests.

When I dump the memory map during one of these events, I see plenty of total free space, but it's scattered in small holes of a few MBs each. Our current monolithic KV cache allocator attempts to allocate a contiguous chunk of memory for a request's full context length. As requests finish and free their space, they leave gaps, resulting in severe external fragmentation. 

Additionally, our prefix caching hit rate metric reads 0% in production, even though users frequently send identical system prompts. The monolithic buffer makes it impossible to share prefixes efficiently, because the entire sequence is tied to a single contiguous memory allocation.

We need to fix this by implementing a block-table allocator. By breaking memory into fixed-size blocks, any free block can satisfy any allocation demand, giving us zero external fragmentation. We also need a routine to analyze our sequence lengths to find the optimal block size that minimizes internal fragmentation, and a lightweight simulation to measure the real prefix-cache hit rate we can achieve.
