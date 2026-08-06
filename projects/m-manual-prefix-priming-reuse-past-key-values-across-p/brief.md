During high-throughput production evaluation of our Hugging Face KV cache serving pipeline, our multi-tenant LLM engine began exhibiting severe GPU memory bloat and state corruption under heavy prompt-sharing workloads.

When multiple incoming client requests share a large common system prompt prefix, the serving engine currently re-executes attention prefill for the shared prefix across every single prompt request. This redundant calculation leads to massive memory duplication across KV cache buffers and creates high time-to-first-token (TTFT) latency spikes.

Furthermore, when attempting to perform context window truncation or step rollbacks by cropping active `DynamicCache` instances, multi-layer KV cache structures become desynchronized or drop key states unpredictably. As a result, token generation fails or produces corrupted text when resumed after context slicing.

Finally, the infrastructure team lacks a deterministic benchmark comparison tool to evaluate memory efficiency and dynamic growth characteristics across `dynamic`, `static`, `offloaded`, and `quantized` KV cache implementation strategies.

We need a standardized implementation in `prefixcache` that:
1. Prefills a shared prefix into a reusable `DynamicCache` and primes independent prompt streams without mutating the shared prefix state.
2. Generates a cache implementation benchmark table for `dynamic`, `static`, `offloaded`, and `quantized` cache strategies.
3. Implements crop and slice operations on `DynamicCache` so generation can resume seamlessly after context trimming.
4. Includes a suite of regression tests catching cache state corruption upon multi-layer context cropping.
