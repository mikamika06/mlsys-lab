# Diagnostic and Reproduction Utilities for ZeRO-1 Memory Partitioning

## Symptom
During mixed-precision training across our distributed GPU clusters, job schedulers are encountering out-of-memory (OOM) errors during the optimizer step despite setting configuration flags for ZeRO Stage 1 optimizer state partitioning. The automated memory estimation tool used by the launcher reports inconsistent headroom estimates compared to actual runtime allocations on multi-node runs.

Upon initial investigation, two issues have surfaced in our internal tooling:
1. The static memory calculator incorrectly models the per-rank optimizer state distribution when scaling world sizes under mixed precision (FP16 weights/gradients with FP32 master weights, momentum, and variance).
2. The distributed wrapper used for prototype local testing fails to properly partition parameter shards across ranks, causing full optimizer state replication instead of sharding $1/N$-th of the states to each rank.

Furthermore, automated parsing scripts designed to extract allocation metrics from standard DeepSpeed startup logs are crashing or failing to extract the ZeRO stage and optimizer partition configuration correctly.

## Deliverables
You must implement a diagnostic package `zero1` that addresses these issues:
1. `zero1/memory.py`: A memory estimation formula calculator that computes per-rank parameter, gradient, and optimizer state memory requirements under standard data parallel and ZeRO-1 partitioned modes.
2. `zero1/log_parser.py`: A startup log parsing tool that extracts ZeRO initialization parameters and estimated memory reduction ratios from log streams.
3. `zero1/distributed.py`: A functional prototype `ToyZeRO1Optimizer` that shards FP32 master parameters, momentum, and variance vectors across ranks, performs local updates, and gathers updated FP16 parameters across processes.
4. `tests/test_regression.py`: A test suite verifying memory sharding invariants and ensuring optimizer updates correctly update parameters across ranks.
