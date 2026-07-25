// Fixed: every intermediate value is a local variable (a register),
// never a round trip through `scratch`. Same 6 arithmetic steps, same
// result, but the only global traffic left is the one input read and
// the one output write per thread.
__global__ void compute_expr(const float* x, float* scratch, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xv = x[i];
        float t1 = xv + 1.0;
        float t2 = xv + 2.0;
        float t3 = t1 * t2;
        float t4 = xv + 3.0;
        float t5 = t3 - t4;
        float t6 = xv + 4.0;
        out[i] = t5 / t6;
    }
}
