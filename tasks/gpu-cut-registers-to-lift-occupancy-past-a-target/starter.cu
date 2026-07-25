// BUG: computes out[i] correctly, but names a separate local variable for
// every intermediate deviation and every squared term instead of
// recomputing into one running accumulator -- 10 live local variables
// instead of 2, tanking modeled occupancy.
__global__ void sum_sq_dev(float* out, const float* x, float a, float b, float c, float d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d0 = x[i] - a;
        float d1 = x[i] - b;
        float d2 = x[i] - c;
        float d3 = x[i] - d;
        float s0 = d0 * d0;
        float s1 = d1 * d1;
        float s2 = d2 * d2;
        float s3 = d3 * d3;
        float total = s0 + s1 + s2 + s3;
        out[i] = total;
    }
}
