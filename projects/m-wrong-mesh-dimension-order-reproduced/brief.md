# Brief: Debug and Optimize 2D Device Mesh Orders for Distributed Training

## Symptom
During multi-node LLM training using 2D parallel mesh topologies (such as Hybrid Sharded Data Parallel / HSDP), jobs report severely degraded training throughput, high collective communication latency across inter-node boundaries, or illegal memory layout/collective mismatched rank errors when initializing distributed process groups.

Monitoring shows heavy AllReduce traffic over slow inter-node links (e.g., WAN/Enet) while high-bandwidth intra-node interconnects (NVLink/NVSwitch) sit underutilized. Furthermore, placing HSDP dimensions in the wrong order causes sub-optimal collective grouping and unexpected rank stride assignments.

## Goal
You must implement a high-performance device mesh topology planner and verifier that correctly constructs 2D Device Mesh mappings, detects wrong mesh dimension orders, verifies HSDP 2D meshes against hardware topologies, and performs optimal fast-axis assignment under heterogeneous network bandwidths.

1. Implement core mesh construction and validation logic to detect and fix inverted axis dimension ordering.
2. Verify HSDP placement properties and ensure local intra-node shard groups map to high-bandwidth fast axes.
3. Construct an automatic topology planner that binds mesh axes to network tiers to maximize communication bandwidth.
4. Write a regression test suite in `tests/test_regression.py` that verifies 2D mesh layout invariants and catches inverted axis orders or topology mismatches.
