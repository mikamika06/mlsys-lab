// Reference: transpose an n x n matrix (B = A^T, both row-major) with
// BOTH the global read and the global write coalesced.
//
// Reading A row-major with (row, col) decomposed so `col` varies fastest
// across the warp is naturally coalesced. But B's element at the
// TRANSPOSED position (col, row) means writing B directly from that same
// thread mapping needs address col*n+row, which is stride-n across the
// warp -- one 128-byte transaction per lane instead of one per warp. Stage
// through __shared__: read A and store the tile with the coalesced-read
// mapping, sync, then read the tile back TRANSPOSED and write B with the
// coalesced-write mapping.
__global__ void tiled_transpose(float* B, const float* A, int n) {
    __shared__ float tile[256];
    int tid = threadIdx.x;

    // Read A coalesced (col fastest matches A's row-major layout),
    // store the tile in that same, un-transposed order.
    int row = tid / n;
    int col = tid % n;
    tile[row * n + col] = A[row * n + col];
    __syncthreads();

    // Write B coalesced (col2 fastest matches B's row-major layout);
    // read the transposed element back out of the tile instead of A.
    int row2 = tid / n;
    int col2 = tid % n;
    B[row2 * n + col2] = tile[col2 * n + row2];
}
