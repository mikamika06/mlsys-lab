// Transpose the last two axes of a (B,H,S,D) tensor into (B,H,D,S).
// One block per (b,h) slice (blockIdx.x in [0, B*H)), S*D threads per
// block. Bounce the tile through __shared__ memory so BOTH the read
// from `in` and the write to `out` are coalesced (see task.md).
__global__ void transpose_bhsd(float* out, const float* in, int B, int H, int S, int D) {
    // TODO
}
