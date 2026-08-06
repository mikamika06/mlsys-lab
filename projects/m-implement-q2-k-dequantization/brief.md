# Q-Quant Superblock Dequantization & Bit-Rate Analysis

We are seeing significant accuracy degradation and incorrect memory allocation when serving GGUF models that rely on K-quantization superblocks (specifically `Q2_K` and `Q3_K`). In our internal model-evaluation pipeline, outputs from dequantized `Q2_K` tensors do not align with reference outputs, and memory planning logic underestimates the storage footprint of quantized weights across the K-quant family.

Investigation points to two separate low-level kernel defects and an incorrect bit-rate calculation:
1. The dequantization loop for `Q2_K` appears to apply scale and offset factors incorrectly across the 16-element sub-blocks, or fails to properly extract the 2-bit quants from the packed byte array.
2. The `Q3_K` unpacker fails to correctly reconstruct the 8-bit scale factors from the packed lower/upper split representation (`hmask`), resulting in corrupted scales for sub-blocks.
3. The theoretical memory model for K-quant superblocks miscalculates total bits-per-weight (bpw), leading to improper KV-cache and weight buffer allocation limits.

Your goal is to complete the low-level dequantization utilities, fix the `Q3_K` high-bit scale reconstruction logic, implement exact `bpw` calculation functions for all standard K-quant formats, and add unit tests in `tests/test_regression.py` that guard against broken dequantization steps and scale corruption.
