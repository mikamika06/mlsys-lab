// Compute out[row][col] = mat[row][col] + vec[col] for a `rows` x `cols`
// matrix `mat` and a length-`cols` vector `vec`, one thread per output
// element (flat index i = row * cols + col). Derive row/col from i
// yourself -- no library broadcasting exists here.
__global__ void broadcast_add(float* out, const float* mat, const float* vec, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: guard i < rows*cols, compute col = i % cols, then
    // out[i] = mat[i] + vec[col];
}
