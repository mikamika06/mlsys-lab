# Incident Report: Unpredictable Model Evictions and High Cold-Load Latencies on Local Runner Nodes

During high-concurrency serving workloads on our T1 local runner nodes, operators have observed sudden drops in active model availability followed by unexpected latency spikes. Specifically, when multiple models contend for limited host RAM and VRAM, the runner control plane evicts models under memory pressure, but the exact sequence in which models are evicted is obscured in raw logs.

Furthermore, operators attempting to manually flush resources via API commands report ambiguity in confirming whether a model has been fully unloaded from memory or if residual mappings persist. Compounding this, subsequent cold loads of models that leverage memory-mapped (`mmap`) weights are occasionally exhibiting disk read throughput profiles identical to initial loads, suggesting that page cache residency optimizations are not being properly leveraged or verified across restarts.

We need robust programmatic tooling to:
1. Parse a sequential time-series of `/api/ps` snapshots to deterministically reconstruct the chronological eviction order of models.
2. Issue an immediate programmatic unload command and verify via subsequent `/api/ps` state inspection that the model is entirely removed.
3. Validate that using `mmap` for model weight loading results in significantly lower physical disk reads on a second cold load compared to the first load due to OS page caching.
