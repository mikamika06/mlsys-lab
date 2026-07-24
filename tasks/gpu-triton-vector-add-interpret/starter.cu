// Compute out[i] = a[i] + b[i] for i in [0, n), one element per thread.
// Keep the access coalesced: thread i must touch address i in a, b, and out.
__global__ void vecAdd(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then out[i] = a[i] + b[i];
}
