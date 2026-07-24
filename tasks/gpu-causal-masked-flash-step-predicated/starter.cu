// Implement one query per lane (s = 32 = one warp, d = 4), streaming
// online-softmax causal attention: keep a running max m_i, denominator
// l_i, and per-dimension numerator, rescaling whenever the max moves.
// Apply the causal mask (j <= i) by PREDICATION -- fold a 0/1 weight into
// the score -- never an early return/continue, so every lane issues the
// same number of loads/stores and the warp never diverges. d = 4 is
// fixed, so unroll the per-dimension accumulators into named scalars
// (this CUDA-C subset has no local arrays).
__global__ void flash_step(const float* q, const float* k, const float* v, float* o,
                            int s, int d, float scale) {
    int i = threadIdx.x;
    // TODO: streaming online-softmax causal attention, predicated.
    o[i * d + 0] = 0.0;
    o[i * d + 1] = 0.0;
    o[i * d + 2] = 0.0;
    o[i * d + 3] = 0.0;
}
