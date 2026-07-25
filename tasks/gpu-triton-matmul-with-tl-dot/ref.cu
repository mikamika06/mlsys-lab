// Reference: block-programmed matmul (the Triton tl.dot model), C = A x
// B, for M=N=K=5 -- NOT a multiple of the BLOCK=4 tile size, so the
// launch necessarily overhangs the matrix on both edges. Grid is a 2x2
// arrangement of BLOCK x BLOCK tiles (flattened 1D: blockRow =
// blockIdx.x/2, blockCol = blockIdx.x%2), each block 16 threads
// (flattened: tx = threadIdx.x%4, ty = threadIdx.x/4). Every thread
// computes its own (row, col), but only WRITES if both are in bounds --
// the boundary tiles' overhanging threads (row >= M or col >= N) are
// masked off instead of reading or writing out of range.
__global__ void block_matmul_masked(float* C, const float* A, const float* B, int M, int N, int K) {
    int tx = threadIdx.x % 4;
    int ty = threadIdx.x / 4;
    int blockRow = blockIdx.x / 2;
    int blockCol = blockIdx.x % 2;
    int row = blockRow * 4 + ty;
    int col = blockCol * 4 + tx;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc = acc + A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
