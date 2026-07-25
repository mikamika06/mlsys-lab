// Reference: one block per row, one thread per feature. Fuses three
// stages into a single kernel / single pass over the row:
//   1) deterministic fixed-seed dropout: keep/drop decided by a pure
//      arithmetic hash of (flat index, seed) -- no RNG call exists in
//      this language subset, so "fixed seed" means "the exact same
//      formula, every time, everywhere" -- kept elements are rescaled by
//      1/(1-p) (inverted dropout), dropped elements become 0.
//   2) residual add: + residual[idx].
//   3) LayerNorm over the row: subtract the row mean, divide by the row
//      stdev (+eps), scale by gamma, shift by beta.
// Only 2 __syncthreads()-separated reductions (mean, then variance) ever
// touch global memory's row data through shared memory -- nothing here
// writes an intermediate stage back to global memory to be read again by
// a second kernel.
__global__ void fused_block(float* out, const float* x, const float* residual,
                             const float* gamma, const float* beta,
                             int d, float dropout_p, float eps, int seed) {
    __shared__ float buf[64];
    __shared__ float sqbuf[64];
    __shared__ float mean_s[1];
    __shared__ float var_s[1];

    int row = blockIdx.x;
    int j = threadIdx.x;
    int idx = row * d + j;

    int h = (idx * 31 + seed * 7 + 11) % 100;
    float thresh = dropout_p * 100.0f;
    float keep = 1.0f;
    if (h < thresh) {
        keep = 0.0f;
    }
    float scale = 1.0f / (1.0f - dropout_p);
    float v = x[idx] * keep * scale + residual[idx];
    buf[j] = v;
    __syncthreads();

    if (j == 0) {
        float s = 0.0f;
        for (int k = 0; k < d; k = k + 1) {
            s = s + buf[k];
        }
        mean_s[0] = s / d;
    }
    __syncthreads();

    float dev = buf[j] - mean_s[0];
    sqbuf[j] = dev * dev;
    __syncthreads();

    if (j == 0) {
        float s = 0.0f;
        for (int k = 0; k < d; k = k + 1) {
            s = s + sqbuf[k];
        }
        var_s[0] = s / d;
    }
    __syncthreads();

    out[idx] = dev / sqrtf(var_s[0] + eps) * gamma[j] + beta[j];
}
