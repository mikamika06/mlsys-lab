# Incident Report: Tail-Latency SLO Violations in vLLM Serving Cluster

## Symptom
Our online serving infrastructure running a large language model is experiencing intermittent spikes in request end-to-end latency, causing a significant fraction of requests to breach their strict Service Level Objective (SLO) deadline. While overall throughput appears stable, downstream clients report frequent timeouts and degraded quality of service during peak traffic intervals. 

Currently, our metrics dashboard only exposes aggregate request latencies and total request counts. When an SLO violation occurs, engineers cannot determine whether the bottleneck stems from excessive time spent waiting in the scheduling queue before execution begins, disproportionately large prompt processing phases (long prefill times due to heavy context lengths), or prolonged token generation phases (long output sequences). Without a precise classification of the root cause for each violating request, capacity planning and scheduling policy tuning remain entirely guesswork.

## Task
Implement the request analysis and classification utilities in `slo/classifier.py` to identify SLO-violating requests and automatically categorize their primary performance bottlenecks into queueing delays, long prefill phases, or long output generation phases. Additionally, write a regression test suite in `tests/test_regression.py` that validates your classification logic against edge cases.
