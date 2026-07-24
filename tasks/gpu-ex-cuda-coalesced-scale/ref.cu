// Reference: coalesced elementwise scale, out[i] = s * in[i].
// Thread i touches address i (in) and address i (out) -> consecutive
// addresses within a warp -> coalesces into one 128-byte transaction
// per access step.
__global__ void scale(float* out, const float* in, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = s * in[i];
    }
}
