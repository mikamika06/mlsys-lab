// Compute out[i] = max(0, (a[i] + b[i]) * c[i]) for i in [0, n), one
// element per thread. Keep every intermediate value in a local scalar
// (a register) -- read each of a[i], b[i], c[i] ONCE and write out[i]
// ONCE. Do not round-trip an intermediate result through `out` (or any
// other global array) between steps.
__global__ void fuse(float* out, const float* a, const float* b, const float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < n, then compute the fused add -> mul -> relu chain
    // using local scalars, and write the result to out[i] once.
}
