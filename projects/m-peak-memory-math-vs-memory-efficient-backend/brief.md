An investigative task has landed on your desk regarding a production regression in a high-throughput transformer inference service using PyTorch's scaled dot product attention. During a large batch rollout, memory consumption on our GPUs spiked significantly higher than expected when switching from a math fallback to a memory-efficient backend under specific masking conditions.

The service relies on PyTorch's scaled dot product attention dispatch mechanism, but operators are reporting that peak memory scales poorly or behaves unexpectedly when dealing with dense attention vs. flash-attention code paths, particularly when `is_causal=True` is provided versus an explicit triangular mask tensor.

Your task is to implement a suite of diagnostic and optimization utilities in `attnlab` to understand and control this behavior. Specifically, you need to:
1. Model and compare peak memory consumption between math and memory-efficient/flash attention backends under varying sequence lengths and batch sizes, verifying that memory usage respects expected scaling limits.
2. Analyze attention FLOPs and HBM traffic, contrasting the theoretical memory footprint of naive materialization of attention weights against flash attention block-wise reduction strategies.
3. Handle mask semantics robustly by correctly mapping and evaluating whether using `is_causal` vs. an explicit triangular tensor affects kernel dispatch, memory overhead, and numerical alignment, and write a robust regression test suite that catches any mishandled fallback or mask representation bugs.

You must fill out the implementation files under `attnlab/` and supply a regression test under `tests/test_regression.py` that fails if fallback behaviors or memory limits are improperly computed.
