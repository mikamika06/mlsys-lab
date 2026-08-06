Ticket ID: TICKET-9482
Title: Multi-replica vLLM cluster suffers from low KV cache hit rates and severe load imbalance under skewed prefix distributions

Symptoms:
Our multi-replica vLLM serving cluster experiences suboptimal KV cache reuse when handling chat and multi-turn request workloads. Standard load-balancing policies such as round-robin and least-outstanding requests distribute traffic evenly across replicas, but they completely ignore the prefix token overlap between incoming prompts and the blocks already resident in each replica's KV cache manager. Consequently, identical or heavily overlapping system prompts and conversation histories are repeatedly recomputed across different instances, driving up time-to-first-token (TTFT) latency and wasting precious GPU memory bandwidth.

Attempts to introduce naive prefix-based hash routing have failed because they create severe hot-spotting and tail-latency explosions. When a specific system prompt or popular prefix receives a burst of traffic, strict prefix routing sends all those requests to a single replica, overwhelming its queue while other replicas sit idle. Furthermore, engineering teams currently lack a unified simulation harness to quantitatively compare round-robin, least-outstanding, and prefix-affinity policies on deterministic trace files. Finally, our capacity planning models are overly simplistic, failing to compute the exact number of replicas required to sustain a given Poisson arrival rate lambda while bounding the target waiting queue depth and queueing delay.

We need a robust, cache-aware routing module implementing prefix hash affinity coupled with a dynamic load-imbalance guardrail, a comprehensive multi-policy trace simulator, and precise queueing-theoretic capacity math.
