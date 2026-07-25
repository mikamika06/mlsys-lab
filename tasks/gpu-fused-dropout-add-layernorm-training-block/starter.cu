// Fuse dropout + residual-add + LayerNorm into ONE kernel: one block per
// row (blockIdx.x), one thread per feature (threadIdx.x, d threads).
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
    // TODO:
    // 1) h = (idx*31 + seed*7 + 11) % 100; keep = (h < dropout_p*100) ? 0 : 1
    //    (inverted dropout: kept values scaled by 1/(1-dropout_p)).
    // 2) v = x[idx]*keep*scale + residual[idx]; buf[j] = v; __syncthreads();
    // 3) thread 0 reduces buf[] into mean_s[0]; __syncthreads().
    // 4) dev = buf[j]-mean_s[0]; sqbuf[j] = dev*dev; __syncthreads();
    //    thread 0 reduces sqbuf[] into var_s[0]; __syncthreads().
    // 5) out[idx] = dev / sqrtf(var_s[0]+eps) * gamma[j] + beta[j];
}
