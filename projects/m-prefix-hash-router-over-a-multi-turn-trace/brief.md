We are observing severe latency spikes in our production LLM inference cluster when serving multi-turn conversational chat workflows. While average prompt processing throughput appears nominal under light traffic, p95 Time-To-First-Token (TTFT) degrades sharply as concurrent chat sessions scale up.

Internal trace analyses show that users participating in extended multi-turn conversations experience erratic delays between turns. In multi-turn chat interactions, the prompt sequence of each turn contains the entire conversation history from preceding turns. However, our frontend load balancer routes incoming request turns across the worker pool without accounting for existing worker KV cache state. Consequently, prefix prompt context that was already computed and cached on worker $A$ during turn $N$ is repeatedly re-computed from scratch when turn $N+1$ lands on worker $B$.

Naive affinity routing strategies introduce a different failure mode: hot multi-turn sessions frequently pile onto a single worker, creating severe queueing delays that far outweigh any prefill speedups gained from cache hits.

We need a flexible routing library that:
1. Computes deterministic token block hashes and tracks worker prefix cache states over multi-turn traces.
2. Simulates and compares execution latency across Round-Robin, Prefix-Hash, and KV-Aware (locality-vs-load) routing policies.
3. Performs locality-vs-load weight tuning to minimize p95 TTFT on representative multi-turn workload traces.
