# Kernel-Eligibility Function & Backend Dispatch

Production inference runs for our flagship model have suffered a sudden latency spike following recent quantization config tweaks and backend dispatch updates. Profilers show that several core matrix-multiplication operations are unexpectedly falling back to slow unoptimized GEMM kernels instead of executing on fused, high-performance hardware kernels (such as Tensor Core FP16/INT8 or specialized AWQ/GPTQ kernels).

Your goal in this exercise is to build and diagnose the kernel-eligibility and dispatch selection mechanism used by our execution engine.

First, you must implement the kernel eligibility function (`is_eligible`) and dispatch selector (`dispatch_kernel`) in `dispatch/selector.py`. The selector evaluates input layer configurations (tensor shapes, data types, quantization schemes, group sizes, and alignment) against registered backend kernel constraints to select the fastest eligible kernel.

Second, you will analyze 20 operational checkpoints captured during model execution traces. You must label which kernel was actually dispatched for each checkpoint, identify which checkpoints fell back to slow paths, and determine the exact minimal configuration changes needed to restore high-performance execution on the fast paths.

Third, you will author regression tests in `tests/test_regression.py` that guard against improper fallback handling and ensure kernel dispatch invariants hold when configurations are updated.
