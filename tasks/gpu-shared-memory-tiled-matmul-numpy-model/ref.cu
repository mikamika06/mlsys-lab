// Reference: classic shared-memory-tiled GEMM, C = A x B, M=N=K=8,
// TILE=4. Grid is 4 blocks (a 2x2 arrangement of output tiles, flattened
// 1D: blockRow = blockIdx.x/2, blockCol = blockIdx.x%2); each block has
// 16 threads (a TILE x TILE arrangement, flattened: tx = threadIdx.x%4,
// ty = threadIdx.x/4) computing one TILE x TILE output tile.
//
// For each of K/TILE=2 K-sub-tiles: every thread cooperatively loads ONE
// element of A's tile and ONE element of B's tile into shared memory,
// barriers, does its TILE-deep partial dot product entirely out of
// shared memory (reusing every loaded element TILE times across the
// tile's threads instead of re-reading global memory), barriers again,
// then moves to the next K-sub-tile. Only after all K-sub-tiles does it
// write its one output element.
__global__ void tiled_matmul(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float As[16];
    __shared__ float Bs[16];
    int tx = threadIdx.x % 4;
    int ty = threadIdx.x / 4;
    int blockRow = blockIdx.x / 2;
    int blockCol = blockIdx.x % 2;
    int row = blockRow * 4 + ty;
    int col = blockCol * 4 + tx;
    float acc = 0.0f;
    int numTiles = K / 4;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * 4 + tx;
        int bRow = t * 4 + ty;
        As[ty * 4 + tx] = A[row * K + aCol];
        Bs[ty * 4 + tx] = B[bRow * N + col];
        __syncthreads();
        for (int k = 0; k < 4; k++) {
            acc = acc + As[ty * 4 + k] * Bs[k * 4 + tx];
        }
        __syncthreads();
    }
    C[row * N + col] = acc;
}
