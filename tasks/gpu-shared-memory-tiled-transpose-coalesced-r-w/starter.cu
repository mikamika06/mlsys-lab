// Transpose an n x n row-major matrix: B = A^T.
//
// This version reads A with the mapping that keeps the READ coalesced
// (col fastest, matching A's row-major layout) and writes straight to
// B's transposed position from that same mapping -- correct, but the
// write address (col*n+row) is stride-n across the warp: one transaction
// per lane instead of one per warp. Fix it by staging through
// __shared__ so both the read from A and the write to B can each use the
// mapping THEY want.
__global__ void tiled_transpose(float* B, const float* A, int n) {
    int tid = threadIdx.x;
    int row = tid / n;
    int col = tid % n;
    B[col * n + row] = A[row * n + col];
}
