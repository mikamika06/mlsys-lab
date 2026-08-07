# Paged KV Block Allocation and Throughput Analysis

## Symptom
Under heavy concurrent load, our TensorRT-LLM serving pipeline exhibits severe memory fragmentation and unpredictable throughput degradation. Traces indicate that request-rate spikes lead to sudden KV cache memory allocation failures, forcing early context evictions and sharp drops in token generation speed.

## Task
Implement a deterministic paged KV cache block planner, a benchmark throughput reporting harness that mimics `trtllm-bench`, and an automated request-rate sweep harness to evaluate serving stability under varying concurrency loads.

## Details
1. **Paged KV Block Planner (`kvplan/planner.py`)**:
   - Implement `calculate_paged_kv_plan(seq_lens, block_size, page_budget)` to determine total blocks allocated, active pages per sequence, wasted/fragmented slots, and memory efficiency ratios.
   - Implement `simulate_block_allocation(request_arrival_pattern, block_size, page_budget)` to track block allocation and deallocation dynamically across request lifecycles.

2. **Benchmark Throughput Reporting (`kvplan/bench.py`)**:
   - Implement `generate_throughput_report(requests_completed, total_prompt_tokens, total_gen_tokens, total_time_sec)` returning benchmark metrics including token generation rates, prompt processing rates, and overall throughput ratios against single-stream baselines.

3. **Request-Rate Sweep Harness (`kvplan/sweep.py`)**:
   - Implement `run_request_rate_sweep(rate_list, block_size, page_budget)` to evaluate system performance across varying request arrival frequencies.
   - Produce a summary mapping arrival rates to total sequence throughput and memory efficiency ratios.

4. **Regression Testing (`tests/test_regression.py`)**:
   - Create tests validating KV block bounds, memory safety, non-negative block allocations, and proper throughput scaling under high concurrency.
