// Reference: one thread per row. First pass accumulates sum and sum of
// squares to get mean and (biased) variance in a single sweep; second
// pass normalizes and applies the per-feature affine transform.
__global__ void layernorm_forward(const float* x, const float* gamma, const float* beta,
                                   float* y, int rows, int D, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0;
        float sumsq = 0.0;
        int d = 0;
        while (d < D) {
            float v = x[row * D + d];
            sum = sum + v;
            sumsq = sumsq + v * v;
            d = d + 1;
        }
        float mean = sum / D;
        float var = sumsq / D - mean * mean;
        float invstd = 1.0 / sqrtf(var + eps);

        d = 0;
        while (d < D) {
            float v = x[row * D + d];
            y[row * D + d] = (v - mean) * invstd * gamma[d] + beta[d];
            d = d + 1;
        }
    }
}
