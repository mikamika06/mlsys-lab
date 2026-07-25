// TODO: one thread per row. Sweep the row once to accumulate sum and
// sum-of-squares, derive mean and (biased) variance from them, then
// sweep the row again to write y[row][d] = (x[row][d]-mean) * invstd *
// gamma[d] + beta[d], where invstd = 1/sqrt(var+eps). See ref.cu.
__global__ void layernorm_forward(const float* x, const float* gamma, const float* beta,
                                   float* y, int rows, int D, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        // your code here
    }
}
