# Ticket: Unstable Inferencing and Memory Bloat Under ORT CUDA Graph Capture

## Symptom
Production instances running our model under ONNX Runtime with CUDA Graph capture enabled are intermittently serving stale outputs during batch-size transitions. Additionally, memory monitoring dashboards show unexpected arena expansion and peak RSS spikes during graph warmup and capture phases, even when input tensors remain bounded.

## Context
We are deploying low-latency ONNX models using ORT's CUDA Execution Provider. To eliminate kernel launch overheads, CUDA graph capture is enabled. However, our pipelines mix dynamic input shapes, direct host-to-device IOBinding, and arena-based memory allocations.

During runtime, changing shapes or binding unaligned memory addresses appears to trigger graph invalidation or cause output buffers to retain stale memory from prior captures. Furthermore, the arena allocator's growth pattern during capture does not align with expected peak resident set size (RSS), leading to unpredictable OOM events under multi-stream workloads.

## Task
Diagnose and resolve these CUDA graph capture anomalies by:
1. Building a CUDA-graph legality classifier to reject invalid execution states before capture.
2. Detecting stale output buffers across graph capture cycles and ensuring proper buffer refresh dynamics.
3. Quantifying arena growth versus peak RSS to safely bound allocation headroom during graph instantiation.
4. Authoring a regression test suite that catches subtle state corruption during graph capture.
