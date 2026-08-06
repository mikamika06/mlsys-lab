# Diagnostic Report: Severe Throughput Degradation During Hybrid Offloading

## Problem Statement
Users running local inference workloads on edge hardware report unpredictable text generation performance. Small adjustments to the GPU layer split often result in massive, non-linear throughput drops (a sharp performance cliff). Additionally, attempts to automatically allocate layers to fit within constrained VRAM budgets frequently fail with out-of-memory (OOM) errors or select suboptimal GPU layer counts that underutilize available VRAM while severely slowing down generation.

## Objective
Implement a lightweight profiler and allocation module to model offloading dynamics, select memory-safe GPU offload fractions, and maximize generation throughput.

## System Requirements
You need to complete three core functions across `offload/profiler.py`, `offload/memory.py`, and `offload/planner.py`:

1. **`offload/profiler.py`**:
   - Implement `find_offload_cliff(profiles)`: Analyze execution time across offload fractions $[0.0, 1.0]$. Locate the steep throughput cliff where inter-device transfer overhead dominates execution, returning the threshold offloaded layer fraction where tokens/sec drops significantly below linear scaling.

2. **`offload/memory.py`**:
   - Implement `fit_layers_in_budget(model_config, memory_budget_bytes)`: Given model weights, KV cache parameters, context length, and safety headroom, compute the exact maximum integer number of transformer layers that fit inside a target GPU memory budget without risking OOM.

3. **`offload/planner.py`**:
   - Implement `select_optimal_offload(model_config, memory_budget_bytes, profiles)`: Given profile throughput curves and a memory budget, choose the exact layer split (`num_gpu_layers`) that maximizes throughput (tokens/sec) while strictly adhering to memory limits and staying on the safe side of performance cliffs.

4. **`tests/test_regression.py`**:
   - Implement suite of unit tests to verify offloading invariants, memory safety margins, and cliff identification rules.
