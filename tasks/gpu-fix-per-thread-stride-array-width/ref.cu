// Fixed: thread tid handles flat element tid directly. Consecutive
// threads touch consecutive addresses -- fully coalesced.
__global__ void elementwise_scale(float* out, const float* in, int w, int h) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w * h;
    if (tid < n) {
        out[tid] = 2.0f * in[tid] + 1.0f;
    }
}
