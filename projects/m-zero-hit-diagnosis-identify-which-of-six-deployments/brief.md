An incident ticket has been assigned to investigate anomalies within a large-scale vLLM serving cluster utilizing automatic prefix caching and tenant isolation mechanisms. The production infrastructure runs six distinct model serving deployments under heavy load, featuring varying configurations of block manager parameters, cache_salt keys, and tenant authorization tokens.

Recently, monitoring telemetry flagged severe cache inefficiency and unexpected cache misses across multiple nodes. Specifically, tenants running identical prompt prefixes are experiencing zero prefix reuse, causing latency spikes and cache thrashing. Concurrently, memory pressure alarms indicate that eviction logic under the LRU policy with reference counts is misbehaving, leading to premature eviction of active blocks or failure to reclaim blocks when reference counts reach zero.

Your task is to diagnose and resolve these issues across three milestones:
1. Identify which of the six deployments can never achieve prefix hits due to configuration anomalies or cache_salt isolation rules.
2. Implement and verify the `cache_salt` isolation constraints to prove that two tenants with identical prompts cannot share blocks improperly.
3. Simulate and validate the LRU eviction policy with reference counts under high memory pressure, ensuring correct block lifecycle management and writing a regression test to safeguard against broken eviction logic.
