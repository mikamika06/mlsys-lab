// Block-programmed matmul, C = A x B, M=N=K=5 with a 4x4-thread block
// tile (BLOCK=4) that overhangs the 5x5 matrix on both edges -- mask
// out-of-range threads instead of reading/writing past M or N.
__global__ void block_matmul_masked(float* C, const float* A, const float* B, int M, int N, int K) {
    int tx = threadIdx.x % 4;
    int ty = threadIdx.x / 4;
    int blockRow = blockIdx.x / 2;
    int blockCol = blockIdx.x % 2;
    int row = blockRow * 4 + ty;
    int col = blockCol * 4 + tx;
    // TODO: if (row < M && col < N), accumulate A[row*K+k]*B[k*N+col]
    // for k in [0,K), then C[row*N+col] = acc.
}
