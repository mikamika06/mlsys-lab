# Blockwise INT4 vs. Per-Channel INT8 Quantization Trade-offs

We are preparing an edge deployment pipeline using Core ML weight quantization tools (`coremltools.optimize.coreml`). During lower-bit quantization trials, team members reported mixed results when comparing INT8 per-channel quantization with INT4 blockwise quantization on the same linear weight matrices. Specifically, some models show surprising MSE degradation under INT4 blockwise setups despite smaller block sizes, while others achieve superior compression with minimal loss.

To establish clear guidelines for our model export stack, we need a diagnostic and benchmarking suite. Your task is to implement the core quantization routines for both INT8 per-channel and INT4 blockwise schemes, calculate scale/zero-point parameters, measure reconstructive mean squared error (MSE), and construct a regression test suite that validates quantization invariants across different block sizes and channel dimensions.

## Requirements

1. **Quantization Routines**: Implement symmetric/asymmetric per-channel INT8 quantization and blockwise INT4 quantization.
2. **MSE Evaluation**: Compute accurate reconstruction loss (MSE) when dequantizing quantized tensors back to FP32, comparing INT8 per-channel against INT4 blockwise schemes on identical weight arrays.
3. **Regression Safety**: Write test cases in `tests/test_regression.py` that catch subtle quantization bugs, such as invalid block boundary handling or improper per-channel scaling calculation.
