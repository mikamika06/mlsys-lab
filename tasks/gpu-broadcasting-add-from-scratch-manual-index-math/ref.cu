// Reference: out[row][col] = mat[row][col] + vec[col], one thread per
// output element, manual row/col index math (no library broadcasting).
// Thread i handles flat index i; col = i % cols is the SAME sequence of
// consecutive integers 0..cols-1 that threadIdx.x sweeps within a row, so
// mat/out/vec all stay coalesced.
__global__ void broadcast_add(float* out, const float* mat, const float* vec, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = rows * cols;
    if (i < n) {
        int col = i % cols;
        out[i] = mat[i] + vec[col];
    }
}
