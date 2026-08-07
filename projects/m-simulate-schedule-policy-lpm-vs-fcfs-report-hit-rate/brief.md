Our vLLM serving instances are experiencing severe cache thrashing on high-concurrency workloads. We currently use a simple First-Come-First-Serve (FCFS) scheduling policy and a basic LRU eviction strategy. With increasing context lengths, this often means that by the time we process a request that shares a long prefix with an earlier prompt, that prefix has already been evicted.

We need to implement and evaluate a simulated cache tree to fix this. First, build a simulator for Longest Prefix Match (LPM) scheduling and compare its hit rate and worst-case wait times against FCFS. LPM should prioritize requests that share the most prefix tokens with the current cache state.

Second, evaluate three different eviction policies when cache capacity is constrained: LRU, LFU, and Longest-Unused-Subtree (LUS). LUS should evict a leaf node and aggressively clear its unused ancestors up to the highest branch point that shares the same access time.

Finally, we are migrating to a HiCache-style tiering architecture. Model a two-tier system where tokens evicted from the GPU cache are moved to host RAM, and cache hits in host RAM promote those tokens back to the GPU. Ensure you can predict the fraction of hits served from each tier.
