# Profiling Speculative Decoding Across Draft/Target Pairs

Our production speculative decoding profiling suite shows inconsistent throughput across our deployed model combinations (`d300m-t7b`, `d1b-t7b`, `d1.5b-t14b`, and `d2b-t70b`). Engineering teams currently configure speculation depth ($\gamma = 5$) statically across all model pairs without measuring the effective cost per accepted token. As a result, several production clusters report severe throughput regressions, while others perform well below expected speedups.

The profiling pipeline lacks a normalized "overhead tax" calculation that accounts for both draft proposal latency, verification overhead, and the empirical token acceptance rate. Without this metric, operators cannot determine whether performance degradation stems from slow draft model execution, high target verification overhead, or poor draft acceptance probabilities.

We need a structured profiling module that reads empirical latency traces and token acceptance probabilities for draft/target pairs, computes cumulative acceptance statistics and overhead tax metrics, generates a consolidated comparison tax table across candidate speculation depths, and identifies the optimal speculation depth $\gamma$ for each pair. Finally, a regression test suite must be provided to ensure tax metric invariants are strictly validated.
