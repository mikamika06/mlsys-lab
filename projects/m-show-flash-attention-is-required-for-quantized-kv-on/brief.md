# Show Flash Attention Is Required for Quantized KV Cache

## Ticket Details
**ID:** `m-show-flash-attention-is-required-for-quantized-kv-on`
**Area:** Local Runners (`rw3-local-runners`)
**Track:** KV Cache Quantization and Flash Attention

## Context & Symptom
Our edge inference runner experiences severe memory bandwidth bottlenecks when scaling context lengths under standard precision. While quantizing the KV cache reduces memory pressure and extends maximum context capacity, standard attention implementations operating on quantized representations incur prohibitive overhead due to un-fused dequantization passes across global memory. Furthermore, aggressive low-bit quantization causes recall degradation at extreme sequence lengths if the dequantization boundary and error bounds are not rigorously managed.

You are tasked with diagnosing and fixing this performance/capacity bottleneck in our local runtime engine (`kvquant`).

## Task Specification
You must implement the core module in `kvquant/`:

1. `kvquant/quant.py`: Implement standard `q8_0` symmetric block-level quantization and dequantization for KV blocks. Ensure dequantization errors are tightly bounded within theoretical absolute tolerances.
2. `kvquant/attention.py`: Implement a memory-efficient fused Flash Attention runtime for `q8_0` quantized KV caches that bypasses intermediate global memory materialization, and provide a context optimizer that selects the optimal context length and KV format given memory and recall constraints.
3. `tests/test_regression.py`: Write comprehensive regression tests that verify quantization precision bounds and validate that standard unfused attention fails to maintain memory/throughput targets relative to fused Flash Attention on quantized KV.

Your tests will be verified by a safeguard harness that injects faulty unfused attention passes and relaxed quantization bounds to ensure safety.
