# Ticket: Discrepancy in Model Benchmark Reports for Quantized Inference Kernels

## Symptom
Our automated performance dashboard is reporting anomalous roofline numbers for low-bit quantization benchmarks across multiple hardware platforms. Engineers noticed that after migrating custom INT4 and FP16 GEMV kernel evaluation suites to our automated profiling pipeline, the reported memory bandwidth efficiency for memory-bound operators frequently exceeds 100% or produces negative speedup predictions relative to theoretical hardware limits. 

Specifically, when running low-batch inference benchmarks on servers equipped with high-bandwidth memory, the predicted analytic memory bandwidth bounds fail to align with empirical profiling measurements. In several benchmark runs for quantized linear layers, the estimated lower bound for execution time does not reflect tensor payload scaling across mixed precision levels (such as INT4 weights paired with FP16 activations). Consequently, optimization decisions based on roofline bottleneck flags are routing memory-bound kernels to compute-optimization pipelines.

## Task
You need to construct an accurate performance measurement module that correctly differentiates between theoretical analytic memory-bandwidth bounds and empirical measured bandwidth. Implement analytic roofline lower-bound calculation routines for arbitrary tensor configurations and precisions, implement empirical bandwidth utilization metrics against profiling traces, and add regression tests that detect incorrect dtype byte accounting or ceiling estimations.
