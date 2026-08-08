# FP8 E4M3 Quantization and Descale Pipeline

Our low-precision inference runtime is dropping accuracy during FP8 linear and attention layer quantization. Inspecting numerical logs shows significant precision degradation across dynamic ranges. We suspect the core FP8 E4M3 encoding/decoding, per-tensor descale reconstruction, and scale search modules are either miscalculating scale factors or misrepresenting the IEEE 754 E4M3 bit-level spec.

Specifically:
- Quantized values do not correctly map to and from the IEEE 754 E4M3 float point representation (1 sign bit, 4 exponent bits, 3 mantissa bits, bias = 7, no infinities, max finite value = 448.0).
- Reconstructed floating-point tensors using per-tensor descale factors deviate excessively from reference floating-point values because maximum absolute values and scaling factors are calculated incorrectly.
- The scale search routine fails to find the optimal scaling factor that minimizes mean squared error (MSE) between original float tensors and their FP8 cast-and-descaled reconstructions across target calibration inputs.

Your task is to fix the core FP8 E4M3 encoding and decoding operations, implement robust per-tensor descale reconstruction, and build a grid-search scaling factor optimizer to bring our low-precision pipeline back to parity.
