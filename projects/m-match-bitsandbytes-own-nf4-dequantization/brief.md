We are currently integrating official bitsandbytes 4-bit NF4 quantized models into our custom inference engine to avoid pulling in massive framework dependencies for edge deployments. However, our pure NumPy fallback for the dequantization step is experiencing unacceptable numerical drift. The downstream model perplexity spikes rapidly, and manual inspection of the restored weight matrices shows a severe discrepancy when compared to reference weights extracted directly from the official library.

The error seems localized entirely in the blockwise NF4 dequantization step. Based on isolated unit tests, the maximum absolute error (`max_abs_err`) between our dequantized blocks and the bitsandbytes targets is consistently non-zero, and in some segments, the values flip entirely in magnitude and sign.

We suspect one of three bugs is present in our pipeline:
1. The 16 floating-point values of the NormalFloat (NF4) quantile codebook are slightly off.
2. The 4-bit nibbles are being unpacked in the wrong order (e.g., extracting the high nibble first instead of the low nibble, causing adjacent weights to swap).
3. The absolute maximum (`absmax`) block scaling is being misaligned against the 64-element block boundaries.

We need you to build a strict `nf4.dequant` module that mirrors bitsandbytes' NF4 dequantization bit-for-bit. Ensure the quantile table matches standard constants, unpack the low nibble first, apply block-wise scaling correctly, and provide a regression test to ensure nibble-swapping does not happen again.
