// Reference: LayerNorm(x + residual) * gamma + beta, FUSED into one
// kernel -- the residual sum is never written to (or re-read from)
// global memory as a separate array. One thread per row (N rows, D
// features): pass 1 accumulates sum and sum-of-squares of x+residual
// (recomputed from x/residual in registers, not stored); pass 2
// recomputes x+residual again and writes the normalized, scaled result.
__global__ void fused_layernorm_residual(float* out, const float* x, const float* residual,
                                          const float* gamma, const float* beta,
                                          int N, int D, float eps) {
    int i = threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        float sumsq = 0.0f;
        for (int d = 0; d < D; d++) {
            float v = x[i * D + d] + residual[i * D + d];
            sum += v;
            sumsq += v * v;
        }
        float mean = sum / D;
        float var = sumsq / D - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < D; d++) {
            float v = x[i * D + d] + residual[i * D + d];
            float norm = (v - mean) * inv_std;
            out[i * D + d] = norm * gamma[d] + beta[d];
        }
    }
}
