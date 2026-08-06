# Group-Wise INT4 Quantization Reconstruction Error Analysis

We are experiencing accuracy degradation in our INT4 inference pipeline for LLMs running on CPU. The quantizer applies symmetric uniform INT4 block/group-wise quantization to weight matrices. However, post-quantization reconstruction errors vary unexpectedly across different layer shapes, block sizes, and activation/weight distribution tails.

Specifically:
- Dequantized weight tensors show high Mean Squared Error (MSE) compared to original FP32 weights under certain group sizes.
- Downstream tasks suffer when weight elements clip excessively or when small group sizes introduce rounding errors near zero.
- We lack an automated framework to classify elements into normal rounding versus saturation-clipping states and measure group-wise reconstruction metrics accurately.

Your task is to implement the group-wise INT4 quantization error calculator and element classifier.

## Requirements & Scope

1. **Quantization & Reconstruction (`quant/group_int4.py`)**:
   - Implement symmetric group-wise INT4 quantization.
   - For a given 1D or 2D tensor and group size $G$, divide data into contiguous blocks of size $G$.
   - Compute scale $S = \frac{\max(|X_{\text{group}}|)}{7}$ per group (clipping to $[-7, 7]$ representation range for signed 4-bit integer values). If $\max(|X_{\text{group}}|) = 0$, set scale $S = 1.0$.
   - Quantize values: $q = \text{clamp}(\text{round}(X / S), -7, 7)$.
   - Dequantize values: $\hat{X} = q \times S$.
   - Calculate group-wise and global Mean Squared Error (MSE) between $X$ and $\hat{X}$.

2. **Element Saturation Classifier (`quant/group_int4.py`)**:
   - Classify each element into one of two categories:
     - `clamped`: elements where $|X / S| > 7.0$ (value saturates INT4 bounds).
     - `in_range`: elements where $|X / S| \le 7.0$ (error dominated by rounding/truncation).
   - Compute metrics breakdown (fraction clipped, clipping error magnitude vs rounding error magnitude).

3. **Regression Safeguard (`tests/test_regression.py`)**:
   - Write unit tests verifying that group-wise reconstruction MSE decreases or remains competitive as group size $G$ shrinks.
   - Ensure tests detect faulty scale calculations (e.g., global scale instead of per-group scale).
