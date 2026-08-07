**TICKET: Evaluate NVFP4 vs MXFP4 for long-tailed weight distributions**

**Symptom:**
We are exploring OCP MXFP4 and NVIDIA's upcoming NVFP4 microscopic scaling formats to aggressively quantize our LLM weights. Currently, our evaluation using MXFP4 shows massive degradation on long-tailed weight distributions. Specifically, when a block of 32 elements contains even a single massive outlier, the 8-bit block scale adjusts to accommodate it, forcing the remaining 31 smaller elements to quantize to zero.

NVFP4 reportedly prevents this zeroing effect by using a two-level scaling approach: a higher-level 8-bit scale for a 256-element "superblock", and a fine-grained 4-bit "vector scale" for every 16 elements.

We need a bit-exact simulation of both quantization strategies to measure overhead and mathematically verify if NVFP4's two-level scale rescues the small weights.

**Tasks:**
1. Implement `round_e2m1` to round elements to the nearest E2M1 magnitude `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]`.
2. Implement `mxfp4` quantization: reshape to blocks of 32, find the block max, calculate the 8-bit scale as `2 ** ceil(log2(max / 6.0))`, scale the block, round, and dequantize.
3. Implement `nvfp4` quantization: reshape to superblocks of 256, find the 8-bit superblock scale. Then, within each superblock, scale down and find a 4-bit vector scale (restricted to powers of 2 from `2**-15` to `2**0`) for each 16-element block. Scale, round, and dequantize.
4. Implement `effective_bits` to compute bits-per-parameter for both formats.
5. Add a regression test to prove that NVFP4 preserves small values when they share a superblock with a large outlier, catching any broken vector-scale logic.
