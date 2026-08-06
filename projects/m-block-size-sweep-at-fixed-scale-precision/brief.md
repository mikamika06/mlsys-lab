# FP4 Microscaling Block-Size Sweep and Quantization Anomalies

Our downstream evaluation pipeline reported unexpected accuracy degradation and quantization error spikes when experimenting with block-size sweeps under fixed scale precision (E8M0). Investigations revealed multiple potential sources of error in our FP4 quantization and unpack routines:

1. Block-size sweeps produce erratic quantization metrics under fixed E8M0 scale representations, but the optimal block size selection routine does not correctly identify the minimal reconstruction error index when evaluating candidate block sizes across tensor slices.
2. Under specific rounding modes or scaling paths, the E8M0 scale quantization exhibits a rounding direction bug where boundary floating-point scale values are inappropriately rounded up or down, corrupting scale factors.
3. Unpacking MXFP4 packed nibble tensors into unpacked low-precision representation fails or shifts nibbles incorrectly, corrupting element-wise reconstruction.

Your task is to fix the packed MXFP4 tensor unpack routines, patch the E8M0 scale rounding logic, and construct a robust block-size sweep routine that correctly identifies the minimal error configuration via `argmin_index`. Finally, you must write regression tests in `tests/test_regression.py` that guard against broken block-size selection logic.
