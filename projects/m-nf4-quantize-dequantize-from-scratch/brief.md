Ticket: Quantization error on normally-distributed weights is surprisingly high

We are trying to write our own 4-bit quantization routines for an experimental branch of our training pipeline, heavily inspired by the NormalFloat4 (NF4) data type from the QLoRA paper. The intent is to maintain a simple, pure-NumPy implementation of blockwise quantization to validate some architectural hypotheses before we commit to writing a custom CUDA kernel.

However, during our tests, the quantization error on normally distributed synthetic data seems suspiciously high. In fact, it is barely better than simple linear INT4 quantization, whereas the authors explicitly proved that the asymmetric NF4 codebook strictly minimizes the quantization error for zero-mean normal distributions.

We suspect the issue stems from one of three places:
1. We are not generating the asymmetric NF4 codebook correctly from standard normal quantiles.
2. There is a bug in how we pack or unpack the 4-bit indices into bytes (we need the first element in the high nibble, and the second in the low nibble).
3. We are failing to properly scale the blocks by their absolute maximum values during the dequantization pass.

We need you to implement the codebook generation correctly, write the blockwise quantize/dequantize routines from scratch, and verify through `compare.py` that NF4 actually outperforms INT4 on normal distributions. Please also add a regression test that guarantees dequantization doesn't silently ignore block scales.
