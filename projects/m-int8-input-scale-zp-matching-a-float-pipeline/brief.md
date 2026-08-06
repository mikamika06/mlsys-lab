# Ticket: Edge Deployment Visual Corruption and Performance Regression

## Symptom
During edge runtime deployment of our vision pipeline on target embedded NPUs, we are observing severe color distortions (red/blue channel swaps) and significant latency overheads. The current deployment relies on an unoptimized pipeline where raw camera frame buffers are transferred as UINT8, converted to FLOAT32 in user-space, normalized via explicit element-wise sub/mul operations, and then quantized using a dynamic scale/zero-point scheme that does not align with the model's trained INT8 quantization parameters.

## Objectives
1. **Int8 Scale/Zero-Point Alignment**: Implement a matching function that maps raw float model quantization metadata to target UINT8/INT8 input tensors, deriving the exact scale and zero-point parameters needed to preserve float parity without clipping errors.
2. **Layout & Color-Order Remediation**: Diagnose and fix channel/layout mismatches between camera capture formats (e.g., NHWC BGR) and model execution contracts (e.g., NCHW RGB).
3. **Graph-Level Normalization Folding**: Fold mean/std normalization and channel transpose ops directly into the exported graph's quantized input stage so the pre-processing executes zero-copy on the accelerator tensor core.
4. **Regression Guard**: Add unit and integration tests under `tests/test_regression.py` validating that the input contract is strictly preserved across various image dimensions and quantization scales.
