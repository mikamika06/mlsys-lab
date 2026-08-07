# Quantize a Tiny Model with GPTQModel End-to-End

## Symptom
When attempting to deploy large language models on memory-constrained edge devices, full-precision 16-bit weights consume excessive VRAM, leading to out-of-memory errors during initialization and inference execution. Although post-training quantization offers a viable reduction path, configuring low-bit quantization parameters (such as bit-width, group size, and dampening factors) manually often results in severe perplexity degradation or runtime compatibility failures when serializing weights using standard GPTQ pipelines. Furthermore, verifying that the quantized output correctly achieves the expected compression ratio without introducing structural corruptions in weight tensors remains cumbersome without automated validation checks.

## Task
Implement an end-to-end quantization configuration and execution utility for a tiny model using custom GPTQ-style quantization logic. You need to build modules that configure quantization knobs, execute weight quantization with proper scale and zero-point computation, and verify compression size ratios. Finally, write robust regression tests to ensure that quantization invariants are strictly maintained across all linear layers.
