# Symptom: Quantized ONNX & TRT Export Instability and Scale Mismatches

During post-training quantization (PTQ) pipeline bring-up for TensorRT and ONNX Runtime execution, model exports demonstrate significant inference accuracy degradation and non-deterministic dynamic range scales. Lower-level calibration dynamic scales are oscillating between runs or failing to clip outlier activations efficiently, causing degraded signal-to-noise ratios in FP8/FP4 tensor paths. Furthermore, hardware-accelerated NVFP4 block-scale packing routines are producing corrupted output tensors during dequantization/requantization round trips across sub-block boundaries.

To stabilize lower-level calibration and ensure exact mathematical reproducibility without reliance on proprietary black-box C++ binaries, you must implement the pure numerical algorithms for max-calibration scale estimation, relative entropy (Kullback-Leibler divergence) calibration scale search, and sub-vector block-scale quantization/dequantization round trips for NVFP4 representations.

## Key Requirements & Functional Boundaries
1. **Max Calibration Scale Search**:
   - Compute running absolute maximum value updates over multi-batch tensor streams.
   - Support symmetric quantization dynamic range scaling for target integer/float formats ($E_{max}$).
   - Compute calibrated scale factors defined as $S = \frac{\text{Amax}}{\text{MaxQuantVal}}$.

2. **Entropy Calibration (KL-Divergence)**:
   - Construct normalized histogram representations of float activation distributions.
   - For candidate clipping thresholds $T$ across histogram bins:
     - Quantize and bin-collapse activation probabilities down to target quantization bin counts (e.g., 128/256 bins).
     - Expand/expand-smooth quantized distributions to retain total probability mass ($P$) vs target representation ($Q$).
     - Calculate KL Divergence $D_{KL}(P \parallel Q) = \sum P(i) \log \left(\frac{P(i)}{Q(i)}\right)$ with small epsilon probability smoothing.
   - Identify optimal dynamic range threshold $T^*$ that minimizes KL divergence.

3. **NVFP4 Sub-Block Packing & Round Trip**:
   - Quantize float 1D/2D arrays into 4-bit floating point representations (E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit) using block-level scaling.
   - Apply sub-block length factors (e.g., block size = 16 or 32 elements per block scale).
   - Perform block max scale extraction, element-wise FP4 codebook quantization, block scale encoding, and bit packing/unpacking round trips.
