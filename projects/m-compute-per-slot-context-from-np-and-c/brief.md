Our production llama-server instances are experiencing critical latency spikes and unexpected Out-Of-Memory (OOM) crashes under multi-tenant production traffic. On-call operators report that tuning the parallel slots flag (-np) alongside total context size (-c) frequently causes context truncation, where individual slots are allocated fewer tokens than required by long prompt payloads. The cluster auto-scaling service currently lacks visibility into how context is partitioned across slots, making it impossible to determine whether increasing -np will improve system concurrency or degrade per-slot capacity below execution thresholds.

Furthermore, automated telemetry checks are failing: metrics scraped from the llama-server /metrics Prometheus exposition endpoint present conflicting counts for processed and total prompt tokens, preventing operators from confirming whether prompt prefix caching is functioning as expected.

We need an internal utility package named llamaslot to resolve these operational issues:
1. Accurately compute per-slot context allocations from total context (-c), parallel slots (-np), and model context limits.
2. Determine the exact -np saturation point where adding more slots drops per-slot context below workload limits or hits hardware thread caps.
3. Parse Prometheus /metrics payloads to evaluate prompt-cache reuse and hit ratios across production instances.
