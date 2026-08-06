# KV Cache Offloading Latency Spikes and Memory Thrashing

## Symptom
Our multi-tenant LLM serving system recently activated two-tier KV cache offloading to manage peak memory pressure on GPU clusters. However, production telemetry shows severe performance degradation for short-to-medium prompt prefix re-use. Requests with prompt lengths between 128 and 2,048 tokens are suffering from tail latency spikes (p99 time-to-first-token increased by 3.4x) when offloading to host RAM and NVMe drives is enabled.

In multiple observed instances, attempting to retrieve and transfer offloaded KV cache blocks across host PCIe or local NVMe storage takes significantly longer than simply executing prompt prefill recomputation directly on the GPU. Furthermore, the storage tier router is indiscriminately spilling small KV prefixes to NVMe and secondary storage tiers, causing disk I/O thrashing and wasting transfer overhead on tiny payloads.

We need an analytical simulator and tier selection framework to accurately predict KV transfer latencies versus GPU recomputation costs, compute exact break-even prefix lengths across hardware targets, and dynamically route workload prefix lengths to the optimal storage tier or prompt recompute path.
