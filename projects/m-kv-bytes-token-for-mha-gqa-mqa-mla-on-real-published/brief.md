# Ticket: KV Cache Footprint Discrepancies Across MHA, GQA, MQA, and MLA Architectures

## Symptom
Long-context inference deployments are experiencing unexpected Out-Of-Memory (OOM) failures and silent memory budgeting errors when serving models utilizing diverse attention mechanisms—ranging from standard Multi-Head Attention (MHA) and Grouped-Query Attention (GQA) to DeepSeek's Multi-head Latent Attention (MLA). Resident memory growth monitoring tools report discrepancies between empirical memory allocation and static theoretical estimates, particularly when handling compressed latent KV representations in MLA.

Without a precise, unified mathematical model and automated validation framework for calculating exact per-token KV memory overheads across published model configurations, system schedulers over-allocate or under-allocate cache blocks, leading to fragmentation or unexpected runtime crashes under high context pressure.

## Objective
Implement a robust analysis module that accurately computes per-token KV cache byte footprints for MHA, GQA, MQA, and MLA architectures, verifies empirical resident memory growth against theoretical derivations, and establishes regression tests to protect against formula regressions.
