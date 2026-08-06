# Inductor Fusion Trace Analysis & Pointwise Scheduler

## Symptom
Our PyTorch Inductor compiler pipeline is logging raw pointwise operations (`abs`, `add`, `relu`, `mul`), but down the line our execution profiler shows memory bandwidth spikes and excessive intermediate buffer allocations. Engineers report that operations that should be fused into a single C++ or Triton kernel are executing in separate passes, or allocating redundant intermediate memory buffers even when elementwise lifetimes do not overlap.

We need a unified library in `inductorsched` that parses `TORCH_LOGS=fusion` schedule logs, runs a greedy pointwise fusion algorithm over a directed acyclic graph (DAG) of tensor operations, and simulates memory buffer reuse (`inplace_buffers`).

## Requirements
1. **Parse Inductor Log Traces (`inductorsched/trace.py`)**: Parse raw log lines from `TORCH_LOGS=fusion` trace outputs to extract fused node groups versus standalone separate operation nodes.
2. **Greedy Pointwise Fusion Simulator (`inductorsched/fusion.py`)**: Build a scheduler that takes a graph of pointwise/elementwise op nodes and greedily fuses producer-consumer pairs when safe (compatible shape, single-consumer or elementwise dependency), validating output against expected schedule logs.
3. **Memory Savings via Buffer Reuse (`inductorsched/memory.py`)**: Compute exact memory allocation footprints for schedule nodes under `inplace_buffers=False` vs `inplace_buffers=True` using lifetime analysis and buffer aliasing.
4. **Safety Net Regression Testing (`tests/test_regression.py`)**: Write tests verifying that fusion never merges non-pointwise reduction boundaries and that buffer reuse never aliases buffers with overlapping read/write lifetimes.
