# Diagnostic Ticket: Distributed Training Deadlocks and Stalled Comm/Compute Overlap

## Symptom Statement
During multi-node PyTorch DistributedDataParallel (DDP) training runs, our production cluster periodically experiences unrecoverable job hangs where worker processes stall indefinitely at communication barriers. Profiling traces collected across GPU ranks show severe variance in communication overlap efficiency prior to these hangs.

Investigation of execution traces reveals that after dynamic recompilation or dynamic shape triggers, individual ranks appear to register parameters and group gradient buckets in differing sequences. While individual ranks execute compute graphs correctly in isolation, collective operations (AllReduce) stall because bucket boundaries and barrier invocation sequences fail to align across the rank topology. Additionally, our performance monitoring tools lack a deterministic mechanism to calculate compute/communication overlap efficiency and identify bucket layout divergence before a rank lockup occurs.

## Objective
Implement a distributed gradient bucket planning module and trace analyzer in `ddpplan/` that:
1. Constructs deterministic, reverse-topological parameter bucket plans given target memory capacities.
2. Quantifies compute-communication overlap efficiency ratios from rank execution traces.
3. Detects rank-divergent bucket plans to prevent collective operation deadlocks.
4. Includes a regression test suite in `tests/test_regression.py` that flags rank-divergent recompile plans.
