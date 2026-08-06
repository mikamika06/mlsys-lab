Our serving engine is running into severe out-of-memory errors when processing requests with ultra-long sequences. As context windows grow to tens of thousands of tokens, the KV cache footprint expands linearly, completely dominating device memory and limiting our maximum batch size to dangerously low levels.

We have been asked to implement a memory analysis and evaluation utility for KV cache eviction strategies, specifically comparing the memory consumption and perplexity profile of StreamingLLM (attention sinks plus a sliding window) against a full KV cache and naive random eviction.

Currently, our engineers lack a precise tooling module to calculate exact byte-level memory savings across sequence lengths and to visualize or verify perplexity curve trends. Without this derivation, we cannot safely configure memory budgets for long-context production deployments or guarantee that retaining attention sinks prevents the catastrophic perplexity spikes observed under window-only or random eviction policies.

Your task is to build a core analysis module that models memory bounds, calculates exact byte footprints for sink-plus-window versus full caches, and processes validation routines for perplexity curve comparisons across eviction paradigms.
