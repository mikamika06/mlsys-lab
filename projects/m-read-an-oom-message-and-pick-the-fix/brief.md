# Incident Report: Production Pipeline Out-Of-Memory Crashes

**Service:** High-Throughput LLM Inference and Finetuning Cluster
**Severity:** P1 - Frequent Unrecovered Job Failures
**Affected Component:** PyTorch CUDA Caching Allocator & Execution Loop

## Symptoms

Over the past week, several training and inference jobs have abruptly terminated with catastrophic `CUDA out of memory` errors despite aggregate GPU metrics showing seemingly sufficient free memory or low active tensor footprints.

Analysis of the raw crash logs reveals verbose PyTorch OOM diagnostic dumps. A representative trace is shown below:
```

torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.20 GiB (GPU 0; 23.69 GiB total capacity; 14.50 GiB already allocated; 1.05 GiB free; 15.55 GiB reserved in total by PyTorch) If reserved memory is significantly larger than allocated memory try setting max_split_size_mb to avoid fragmentation.

```

Engineering teams observed two distinct anomalies:
1. Jobs fail even when the requested chunk size is smaller than the reported free memory, indicating severe memory fragmentation where large blocks are broken down into unusable small chunks.
2. Long-running training loops exhibit a widening gap between `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`, where memory is held by the caching allocator long after tensors are dropped, eventually triggering false OOMs or preventing subsequent larger allocations.

## Objective

Your task is to build a robust diagnostic and mitigation library (`triage`) that automates OOM log parsing, determines the precise corrective action based on allocator states, simulates caching allocator fragmentation behavior to tune `max_split_size_mb`, and tracks true memory consumption versus reserved overhead during execution loops.

Specifically, you must implement:
1. An OOM message parser that extracts allocation metadata and selects the correct remediation strategy.
2. An allocator optimization routine that evaluates allocation traces to select the optimal `max_split_size_mb` configuration to prevent internal fragmentation.
3. A telemetry tracker that monitors `memory_allocated` vs `memory_reserved` profiles across loop iterations, complete with regression tests to protect allocator invariants.
