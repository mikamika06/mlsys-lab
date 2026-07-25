// Tiled matmul: C = A * B for M x K times K x N, using 16x16 tiles held
// in shared memory. Each block computes one 16x16 tile of C. This
// simulator's CUDA-C frontend is 1D-only (grid/block are ints), so the
// tile's local (tx, ty) and the block's (blockRow, blockCol) must be
// derived by hand from the linear threadIdx.x / blockIdx.x:
//   tx = threadIdx.x % 16, ty = threadIdx.x / 16
//   blocksPerRow = N / 16
//   blockCol = blockIdx.x % blocksPerRow, blockRow = blockIdx.x / blocksPerRow
//   row = blockRow*16 + ty, col = blockCol*16 + tx
// Then, for kt = 0 .. K/16 - 1:
//   - Cooperatively load one 16x16 tile of A and one of B into
//     __shared__ float As[256], Bs[256]: As[ty*16+tx] = A[row*K + kt*16+tx],
//     Bs[ty*16+tx] = B[(kt*16+ty)*N + col].
//   - __syncthreads().
//   - Accumulate acc += As[ty*16+k] * Bs[k*16+tx] for k in [0,16).
//   - __syncthreads().
// Finally C[row*N+col] = acc.
__global__ void matmul_tiled(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float As[256];
    __shared__ float Bs[256];
    int lane = threadIdx.x;
    int tx = lane % 16;
    int ty = lane / 16;
    int blocksPerRow = N / 16;
    int blockCol = blockIdx.x % blocksPerRow;
    int blockRow = blockIdx.x / blocksPerRow;
    int row = blockRow * 16 + ty;
    int col = blockCol * 16 + tx;
    // your code here
}
