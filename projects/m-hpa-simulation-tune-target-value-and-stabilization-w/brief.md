Our new vLLM serving cluster is thrashing wildly. Sometimes it spins up 50 replicas to handle a short burst of requests, but before those instances even finish downloading the model weights, the load drops and it terminates them. Then another burst hits, and the cycle repeats. Users are seeing massive latency spikes because requests are getting routed to cold instances or dropped during aggressive downscaling. We tried increasing the target concurrency per replica, but that just causes requests to queue up and eventually OOM.

Separately, our KV cache hit rates have plummeted since we enabled multi-replica routing. Multi-turn conversations seem to be frequently hitting instances without their prefix cache.

We need to implement proper HPA stabilization and systematically tune our configuration to minimize both thrashing and unmet demand. We also need to quantify exactly how much hit rate we are losing to the random load balancer so we can justify building sticky session affinity.

To simulate the HPA, for each step `t` (0 to N-1):
1. The active replicas at step `t` is `current` (which starts as `initial_replicas`).
2. The raw desired replicas for this step is `max(1, ceil(current * metric[t] / target_value))`.
3. The replicas active for the *next* step (t+1) will be the maximum of the raw desired replicas computed over the last `stabilization_window` steps (inclusive of the current step `t`).

For `tune_hpa`, the cost function for a configuration trace is `churn * 100.0 + deficit` where:
- `churn` is the sum of absolute differences between active replicas in consecutive steps (from `t` to `t-1`).
- `deficit` is the sum over all steps `t` of `max(0.0, metric[t] - target_value * replicas[t])`.
Return the index of the best configuration.

For `quantify_hit_rate_loss`, a session of length K has K turns. The first turn is a cold start (miss). Under perfect session affinity, the remaining K-1 turns are hits. Under random routing, each of the K-1 turns has only a `1 / num_replicas` chance of landing on the same replica as the previous turn. Return the difference between the perfect affinity hit rate and the expected random hit rate across all given sessions.
