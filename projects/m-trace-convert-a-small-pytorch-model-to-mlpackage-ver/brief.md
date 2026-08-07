# coreml-trace-convert: Export and Validate PyTorch Models in Core ML

Production edge deployments on Apple Silicon require converting PyTorch models to Core ML's `.mlpackage` format using `coremltools`. However, subtle numerical mismatches, precision trade-offs, and unsupported PyTorch operators frequently break export pipelines during integration testing.

Your task is to implement and test a complete Core ML export workflow for small PyTorch models:

1. **Trace & Convert**: Build a pipeline that takes a PyTorch module, traces it with sample inputs, converts it to a Core ML model package using `coremltools`, and computes the maximum absolute error (`max_abs_err`) between PyTorch and Core ML output tensors on identical evaluation inputs.
2. **Precision Analysis**: Implement size measuring and conversion routines across `float32` and `float16` compute precisions. Calculate the disk footprint reduction ratio and verify precision-dependent accuracy metrics.
3. **Op-Not-Supported Recovery**: Handle missing operator support by intercepting conversion failures, inspecting torch trace ops, and attempting fallback dynamic tracing/conversions or diagnostic reporting.
4. **Regression Safeguard**: Write test cases in `tests/test_regression.py` that validate numeric tolerance bounds and catch edge-case precision degrades or conversion failures.
