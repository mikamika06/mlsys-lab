# Optimizer Step Overheads: Loop vs Foreach vs Fused Adam

During large model training, monitoring logs revealed that GPU utilization drops significantly during the optimizer step despite high GPU memory bandwidth availability. Profiling indicates that the Adam optimizer step launches thousands of individual CUDA kernels per iteration, dominating the execution time with launch overheads and redundant memory round-trips for small parameters.

You are tasked with analyzing and addressing this optimizer step bottleneck by quantifying kernel launch counts and implementing optimizer dispatch strategies.

## Tasks

1. **Kernel Count Analysis & Multi-Tensor Grouping**
   Implement helper logic to track kernel dispatch overhead across three execution modes: per-parameter loop, PyTorch-style `foreach` multi-tensor application, and fully fused Adam execution. Group model parameters by tensor dtype and device layout into efficient contiguous batches suitable for multi-tensor vectorized operations.

2. **Fused vs Unfused Numerics & Kernel Dispatch Estimation**
   Implement single-tensor Adam (`loop`), batched Adam (`foreach`), and elementwise-fused Adam (`fused`) state updates. Calculate expected CUDA kernel launch counts for a parameter set across all modes, and ensure mathematical equivalence of the parameter updates over multi-step training trajectories within floating-point tolerance bounds.

3. **Regression Test Suite**
   Write unit tests in `tests/test_regression.py` validating that your optimizer implementation correctly groups multi-tensor inputs without dropping parameters or crossing dtype boundaries, and verifies that fused and foreach updates match reference numeric trajectories.
