// Shared-memory-tiled GEMM, C = A x B, M=N=K=8, TILE=4. 4 blocks of 16
// threads each (blockRow=blockIdx.x/2, blockCol=blockIdx.x%2;
// tx=threadIdx.x%4, ty=threadIdx.x/4), one TILE x TILE output tile per
// block. See sol comment in ref.cu for the full per-K-sub-tile
// load/barrier/compute/barrier structure this must follow.
__global__ void tiled_matmul(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float As[16];
    __shared__ float Bs[16];
    int tx = threadIdx.x % 4;
    int ty = threadIdx.x / 4;
    int blockRow = blockIdx.x / 2;
    int blockCol = blockIdx.x % 2;
    int row = blockRow * 4 + ty;
    int col = blockCol * 4 + tx;
    // TODO: acc = 0; for each of K/4 K-sub-tiles: load
    // As[ty*4+tx]=A[row*K + t*4+tx], Bs[ty*4+tx]=B[(t*4+ty)*N + col];
    // barrier; accumulate As[ty*4+k]*Bs[k*4+tx] for k in [0,4); barrier.
    // Then C[row*N+col] = acc.
}
