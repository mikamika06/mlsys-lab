// Fixed: recompute each squared deviation inline into a single running
// accumulator instead of naming every intermediate. Only 2 local
// variables (i, acc) -- everything else is a recomputed expression, never
// a stored register.
__global__ void sum_sq_dev(float* out, const float* x, float a, float b, float c, float d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    if (i < n) {
        acc += (x[i] - a) * (x[i] - a);
        acc += (x[i] - b) * (x[i] - b);
        acc += (x[i] - c) * (x[i] - c);
        acc += (x[i] - d) * (x[i] - d);
        out[i] = acc;
    }
}
