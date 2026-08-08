# Kernel Arithmetic Intensity & Roofline Model Classification

Profiling GPU execution bottlenecks across different matrix and vector operations reveals that some kernels are severely constrained by memory bandwidth, while others are bottlenecked by peak compute performance.

In this exercise, you will implement a roofline modeling utility that analyzes tensor operations to predict roofline ceilings and rank kernels by their operational intensity.

You will build three key modules in `roofline/`:
1. `classify.py`: Given a kernel's Arithmetic Intensity (AI in FLOP/byte) and a target hardware device's ridge point (Peak GFLOP/s / Memory Bandwidth in GB/s), classify the kernel as `"compute-bound"` or `"memory-bound"`, and calculate its roofline-predicted maximum achievable performance in GFLOP/s.
2. `intensity.py`: Compute the exact Arithmetic Intensity (FLOPs divided by total memory bytes accessed) for 6 canonical matrix/tensor kernel shapes:
   - Vector Addition ($Y = A + B$)
   - Matrix-Vector Multiplication ($y = A x$)
   - Matrix-Matrix Multiplication ($C = A B$)
   - Batched Matrix Multiplication ($C = A B$)
   - 2D Convolution ($Y = X \ast K$)
   - Layer Normalization ($Y = \text{LayerNorm}(X)$)
3. Rank a list of configured kernel instances from most memory-bound (lowest AI) to most compute-bound (highest AI).

Finally, you will write a safeguard test suite in `tests/test_regression.py` that verifies kernel classification boundaries and catches improper memory bandwidth scaling in custom roofline predictions.
