# Symptom: Quantization Block Distortion in GGUF Tensor Export

Our quantization pipeline produces corrupted tensor values when exporting legacy 32-element block formats (`Q4_0`, `Q4_1`, and `Q5_1`). While downstream inference runs without crashing, model outputs show high perplexity spikes and near-zero cosine similarity against floating-point baselines.

Internal inspect tools reveal two primary issues:
1. Low-precision scale factors for affine and symmetric 4-bit blocks do not accurately preserve the original dynamic range, leading to severe truncation near zero or extreme clipping.
2. The 5-bit packing routines produce bit-pattern mismatches when unpacking the high-bit (`qh`) plane alongside low-nibble bit-planes, distorting sign and magnitude extraction for odd-indexed elements within each 32-element block.

To restore numerical accuracy across model exports, we need a complete Python reference implementation that correctly packs and unpacks floating-point arrays to and from `Q4_0`, `Q4_1`, and `Q5_1` block structures, adhering strictly to standard GGUF layout specifications.

Your task:
- Implement `quantize_q4_0` and `dequantize_q4_0` in `qblocks/q4_0.py`.
- Implement `quantize_q4_1` and `dequantize_q4_1` in `qblocks/q4_1.py` with proper scale and minimum offset handling.
- Implement `quantize_q5_1` and `dequantize_q5_1` in `qblocks/q5_1.py` including `qh` bit-plane extraction and packing.
- Provide regression tests in `tests/test_regression.py` that guard against broken scale calculations and corrupted bit-plane extractions.
