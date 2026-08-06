We are configuring an autoscaler for our serverless inference deployment. To save costs, the system scales down to zero replicas during periods of inactivity. However, when a sudden burst of traffic arrives, it triggers a "scale from zero" event. The time it takes to spin up a new replica (the cold start latency) means that incoming requests during this window experience significant delays. This is our "cold start exposure."

Your task is to model this behavior and optimize our scale-down policy.

We model the system in discrete time steps (seconds).
The system always starts in the `WARM` state.
- If the system is `WARM` and receives `0` requests, an idle counter starts at `1`. If the counter reaches `idle_timeout`, the system transitions to `COLD`. Otherwise, it transitions to `IDLE`.
- If the system is `IDLE` and receives `0` requests, the counter increments. If it reaches `idle_timeout`, it transitions to `COLD`. If it receives `> 0` requests, it transitions back to `WARM` and the counter resets.
- If the system is `COLD`, it saves money, and for every second it remains in this state with `0` requests, `cold_time_seconds` increments. Any `> 0` requests trigger a scale-up. The system transitions to `WARMING` (or immediately to `WARM` if `cold_start_latency == 1`). The requests received during this tick (and subsequent `WARMING` ticks) are added to our `exposed_requests` count. The warmup counter starts at `1`.
- If the system is `WARMING`, it continues to accumulate `exposed_requests` and increments the warmup counter until it reaches `cold_start_latency`, transitioning to `WARM`.

**Milestone 1**: Implement `simulate_scale_to_zero(traffic, idle_timeout, cold_start_latency)` to return a tuple `(exposed_requests, cold_time_seconds)`. (Assume `idle_timeout >= 1` and `cold_start_latency >= 1`).

**Milestone 2**: Implement `find_optimal_timeout(traffic, cold_start_latency, max_exposure_ratio)`. It should return the **smallest** integer `idle_timeout` (from `1` to `len(traffic)`) such that the ratio of `exposed_requests` to the total requests in the traffic pattern is less than or equal to `max_exposure_ratio`.

**Milestone 3**: Write a test in `tests/test_regression.py` that verifies `find_optimal_timeout` returns the *smallest* valid timeout. We will run your test against a faulty implementation that incorrectly returns the *largest* valid timeout.
