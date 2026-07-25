// BUG: every intermediate value is round-tripped through the `scratch`
// buffer in global memory instead of staying in a local variable --
// exactly what a real compiler emits when it runs out of registers and
// spills to per-thread "local memory" (itself backed by global memory).
// The result is numerically correct, but every extra store/load here
// is extra real global-memory traffic. Fix it: compute the same 6
// arithmetic steps (see ref.cu) using ONLY local variables -- xv, and
// each intermediate term -- so `scratch` is never touched at all.
__global__ void compute_expr(const float* x, float* scratch, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        scratch[i * 4 + 0] = x[i] + 1.0;
        scratch[i * 4 + 1] = x[i] + 2.0;
        scratch[i * 4 + 2] = scratch[i * 4 + 0] * scratch[i * 4 + 1];
        scratch[i * 4 + 3] = x[i] + 3.0;
        float t5 = scratch[i * 4 + 2] - scratch[i * 4 + 3];
        float t6 = x[i] + 4.0;
        out[i] = t5 / t6;
    }
}
