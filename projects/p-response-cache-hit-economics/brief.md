# Ticket: Response Cache Hit Economics

**Symptom:**
After enabling the prompt/response cache in our serving layer, system memory usage jumped significantly, leading to unexpected swap thrashing and degraded tail latency. Despite high expectations, the end-to-end throughput improvements are barely noticeable, and memory pressure is forcing premature cache churn.

**Context:**
The infrastructure team enabled response caching across the fleet to reduce redundant LLM compute. However, without accounting for cache entry sizes, entry lifespans, hit rates, and host memory cost, the net cost-benefit ratio is currently negative. We need a clear, quantitative framework to evaluate when response caching actually yields net cost and latency savings versus when it wastes memory bandwidth and capacity.

**Task:**
1. Parse request traces to measure prompt key repetition and calculate raw hit rates.
2. Build an economic model comparing saved compute FLOPs/latency against memory footprint costs.
3. Quantify memory overheads and cache capacity limits under realistic workloads.
4. Implement eviction policies (LRU, LFU, Cost-Aware) to maximize net economic gain.
5. Provide data-driven recommendations on cache allocation based on work-rate thresholds.
6. Build a robust test suite in `tests/test_regression.py` that catches common misconfigurations and economic edge cases.
