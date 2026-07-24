// Reference: coalesced elementwise vector add, out[i] = a[i] + b[i].
// Thread i touches address i in a, b, and out -> consecutive addresses
// within a warp -> coalesces into one 128-byte transaction per access step.
__global__ void vecAdd(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
