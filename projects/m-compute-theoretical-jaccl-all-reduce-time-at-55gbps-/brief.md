# Ticket: Distributed Inference Performance Degradation on Edge MLX Ring Backend

## Symptom Report
Our edge compute cluster running multi-node MLX inference over Thunderbolt-5 links is experiencing severe throughput degradation when scaling beyond a single machine. Benchmarks on our 2-node and 4-node Apple Silicon clusters reveal that end-to-end token generation latency increases by up to 3.5x compared to linear scaling projections.

System telemetry indicates that worker processes spend an excessive proportion of execution time blocked during inter-node communication. Specifically, during the ring all-reduce phase across 55Gbps Thunderbolt-5 interconnects, processing threads show high waiting times. Furthermore, when attempting pipeline parallel execution to hide latency, GPU compute utilization drops below 45%, with profiling logs indicating large idle gaps (bubbles) between step executions.

We currently lack an analytical model to quantify ring all-reduce transfer times at 55Gbps, measure the precise latency overhead relative to single-process local computation, and calculate the exact microbatch scheduling parameters required to bound pipeline bubble fractions across 2-rank and 4-rank cluster configurations.
