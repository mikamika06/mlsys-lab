// Transpose a `rows` x `cols` row-major matrix `in` into a `cols` x
// `rows` row-major matrix `out`: out[c][r] = in[r][c]. One thread per
// input element (tid = blockIdx.x * blockDim.x + threadIdx.x).
__global__ void naive_transpose(float* out, const float* in, int rows, int cols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    // TODO: guard tid < n, r = tid / cols, c = tid % cols,
    // out[c * rows + r] = in[tid];
}
