// Reference: LayerNorm backward. One thread per (row, col) element
// (blockDim.x = B*D). Since this CUDA-C subset has no atomics and no
// cross-block reduction primitive, every thread recomputes whatever
// per-row or per-column reduction it needs entirely on its own (over
// this task's small B=4, D=8) instead of relying on a shared partial
// sum written by another thread -- redundant arithmetic, but no thread
// ever reads a value another thread hasn't finished computing.
__global__ void layernorm_backward(float* dx, float* dgamma, float* dbeta, const float* dy, const float* x, const float* gamma, int B, int D) {
    int tid = threadIdx.x;
    int row = tid / D;
    int col = tid % D;
    float eps = 0.00001;

    // This row's mean and (biased) variance.
    float mean = 0.0;
    for (int j = 0; j < D; j++) { mean = mean + x[row * D + j]; }
    mean = mean / D;
    float var = 0.0;
    for (int j = 0; j < D; j++) {
        float d = x[row * D + j] - mean;
        var = var + d * d;
    }
    var = var / D;
    float std = sqrtf(var + eps);

    // mean(g) and mean(g * xhat) over this row, g_j = dy_j * gamma_j.
    float mean_g = 0.0;
    float mean_g_xhat = 0.0;
    for (int j = 0; j < D; j++) {
        float xhat_j = (x[row * D + j] - mean) / std;
        float g_j = dy[row * D + j] * gamma[j];
        mean_g = mean_g + g_j;
        mean_g_xhat = mean_g_xhat + g_j * xhat_j;
    }
    mean_g = mean_g / D;
    mean_g_xhat = mean_g_xhat / D;

    float xhat = (x[row * D + col] - mean) / std;
    float g = dy[row * D + col] * gamma[col];
    dx[row * D + col] = (g - mean_g - xhat * mean_g_xhat) / std;

    // dgamma[col] / dbeta[col]: sum over every row. Every thread that
    // shares this column recomputes and writes the SAME value, so
    // there is no write conflict even though several threads target
    // the same output slot.
    float dg = 0.0;
    float db = 0.0;
    for (int r = 0; r < B; r++) {
        float m = 0.0;
        for (int j = 0; j < D; j++) { m = m + x[r * D + j]; }
        m = m / D;
        float v = 0.0;
        for (int j = 0; j < D; j++) {
            float d = x[r * D + j] - m;
            v = v + d * d;
        }
        v = v / D;
        float s = sqrtf(v + eps);
        float xh = (x[r * D + col] - m) / s;
        dg = dg + dy[r * D + col] * xh;
        db = db + dy[r * D + col];
    }
    dgamma[col] = dg;
    dbeta[col] = db;
}
