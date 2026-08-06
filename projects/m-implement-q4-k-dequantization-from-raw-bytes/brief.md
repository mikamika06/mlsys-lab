We are adding direct evaluation of `Q4_K` and `Q6_K` quantized tensors to our Python inference engine. To do this, we need to efficiently unpack their super-block structures directly from raw memory bytes into flat `float` arrays.

In `llama.cpp`'s k-quant system:
- A `Q4_K` block is 144 bytes and decodes to 256 weights. It starts with two float16 values (`d` and `dmin`), followed by 12 bytes of packed 6-bit scales and minimums (8 scales, 8 mins), and ends with 128 bytes of 4-bit quantized weights. The weights are grouped such that each 32-byte chunk holds two 32-weight blocks (lower and upper nibbles).
- A `Q6_K` block is 210 bytes and decodes to 256 weights. It is arranged as 128 bytes of lower 4-bit quants, 64 bytes of upper 2-bit quants, 16 signed 8-bit scales, and one float16 `d`.

Your task is to implement the exact byte-level dequantization for both block types. The tricky part is `unpack_6bit_scales_and_mins` as it distributes the scale/min pairs across non-sequential bits.

Pay close attention to bitwise operations. Ensure you correctly test the signs of the `Q6_K` scales in `tests/test_regression.py`.
