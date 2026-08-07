# Ticket: Runner Reload Cascades and State Bleed in Heterogeneous Serving

We are experiencing severe performance degradation and state leakage during high-concurrency request routing in our local runner cluster. Under mixed workloads involving different model configurations, sequence lengths, and generation parameters, the runner processes are experiencing excessive restarts, and request-level configurations are improperly bleeding into subsequent requests.

## Symptoms Observed

1. **Excessive Runner Restarts:** When dispatching a batch of 12 heterogeneous requests, the total number of runner process restarts spikes dramatically depending on the execution order. Because each distinct parameter set or config alteration can trigger a reload, naive scheduling results in thrashing the local runner engine. We need to find an optimal reordering of these 12 requests to minimize total runner reloads.
2. **Config-Driven Reload Triggers:** The runner cluster behaves inconsistently when various runtime configuration parameters are modified. Across a set of 8 distinct configuration changes, some modifications force an immediate and costly runner restart (re-initializing context buffers and weights or resetting engine states), while others apply dynamically without a restart. We need a reliable mechanism to predict which changes force a restart.
3. **State Leakage / Isolation Failure:** A request-level specification of context window size (`num_ctx`) provided via request parameters is currently persisting across execution boundaries. When a request overrides `num_ctx`, that modified context size improperly carries over to the next incoming request on the same runner session, causing truncated generations or unexpected out-of-memory errors.

## Investigation Goal

Your task is to implement the core modules for `heterogeneous` that:
- Compute the optimal request reordering sequence for 12 heterogeneous requests to minimize total runner reloads.
- Accurately predict which of 8 candidate configuration changes force a runner restart versus those that apply dynamically.
- Enforce strict request boundary isolation so that request-level `num_ctx` configurations do not persist into subsequent requests, accompanied by a robust regression test suite that catches any implementation regression where isolation is broken.
