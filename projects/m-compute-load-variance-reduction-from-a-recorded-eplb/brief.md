# EPLB Load-Variance Reduction and Redundant Expert Allocation

Our Expert Parallelism Load Balancing (EPLB) telemetry shows severe stragglers during long-context Mixture-of-Experts (MoE) decoding phases. Certain ranks experience heavy expert load skew, causing downstream pipeline stalls across nodes. While rebalancing events are recorded, our current reporting pipeline lacks precise measurement of load-variance reduction across physical ranks.

Furthermore, we need to dynamically assign redundant expert replicas to heavily loaded experts to bound the maximum per-rank load, and precisely derive the minimum number of redundant replicas required to meet target load thresholds under tight memory budgets.

Your task is to implement the EPLB analysis and rebalancing utilities:

1. In `eplb/variance.py`, implement `compute_variance_reduction` to record per-rank compute loads before and after rebalancing, computing variance reduction metrics and load statistics across physical ranks.
2. In `eplb/redundant.py`, implement `rebalance_greedy_redundant` to allocate a fixed budget of redundant expert replicas onto physical ranks using a greedy highest-load-first strategy.
3. In `eplb/bounds.py`, implement `min_redundant_replicas` to mathematically derive the minimal number of extra replicas needed so that no rank exceeds a specified target load ceiling under optimal greedy placement.
4. In `tests/test_regression.py`, author unit tests ensuring that `min_redundant_replicas` correctly rejects naive uniform allocation bounds and enforces tight bounds under unbalanced expert work distributions.
