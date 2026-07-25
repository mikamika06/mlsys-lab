// Register-blocked GEMM microkernel. Each thread owns a 2x2 tile of C:
//   tilesPerRow = N / 2
//   tileCol = tid % tilesPerRow, tileRow = tid / tilesPerRow
//   row0 = tileRow*2, col0 = tileCol*2
// Keep 4 running accumulators (acc00, acc01, acc10, acc11) as plain
// local floats, all starting at 0. For k = 0 .. K-1: load
// a0=A[row0*K+k], a1=A[(row0+1)*K+k], b0=B[k*N+col0], b1=B[k*N+col0+1],
// then accumulate the outer product: acc00+=a0*b0, acc01+=a0*b1,
// acc10+=a1*b0, acc11+=a1*b1. Finally write all 4 accumulators to their
// C positions: C[row0*N+col0]=acc00, C[row0*N+col0+1]=acc01,
// C[(row0+1)*N+col0]=acc10, C[(row0+1)*N+col0+1]=acc11.
__global__ void gemm_regblock(float* C, const float* A, const float* B, int M, int N, int K) {
    int tid = threadIdx.x;
    int tilesPerRow = N / 2;
    int tileCol = tid % tilesPerRow;
    int tileRow = tid / tilesPerRow;
    int row0 = tileRow * 2;
    int col0 = tileCol * 2;
    // your code here
}
