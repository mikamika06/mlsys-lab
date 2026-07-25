// TODO: each thread owns a 2x2 output tile: rows (row0, row0+1),
// columns (col0, col0+1), where row0 = (idx/tiles_per_row)*2, col0 =
// (idx%tiles_per_row)*2, tiles_per_row = N/2. At every k, load
// A[row0][k], A[row1][k], B[k][col0], B[k][col1] into local variables
// and accumulate all 4 products (acc00, acc01, acc10, acc11), reusing
// each loaded value across the two outputs that need it. Write the 4
// accumulators to their 4 output positions after the loop. See ref.cu.
__global__ void coarsened_matmul(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tiles_per_row = N / 2;
    if (idx < tiles_per_row * tiles_per_row) {
        int ti = idx / tiles_per_row;
        int tj = idx % tiles_per_row;
        int row0 = ti * 2;
        int col0 = tj * 2;
        C[row0 * N + col0] = 0.0;
    }
}
