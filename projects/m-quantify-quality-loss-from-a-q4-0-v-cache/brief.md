We are upgrading our GGUF inference pipeline in `llama.cpp` style to run multi-sequence parallel slots (`-np`) while keeping the memory footprint under control by using `q4_0` quantization specifically for the V cache (Value cache).

However, high-concurrency serving with parallel slots exhibits non-linear memory growth, and aggressive quantization of the Value cache introduces numerical drift that impacts generation quality. We need a clean internal accounting module that can:
1. Predict multi-slot KV cache allocation sizes across parallel slots (`-np`) with block alignment.
2. Measure quality loss (relative error vs FP16/FP32 baseline) when quantizing the V cache to `q4_0` blocks across dynamic context lengths.
3. Validate through regression tests that quantized V cache evaluations detect quality degradations before deploying to production.

Implement the KV cache sizing and q4_0 V-cache error measurement functions to meet memory and accuracy bounds.
