// Compute a 2-tile matmul (tile_k = K/2), staging each tile through shared
// memory. Overlap the tile-1 prefetch with the tile-0 compute -- don't
// spend an extra __syncthreads() serializing "load tile 1" after "compute
// tile 0" if you don't have to.
__global__ void tiled_matmul_double_buffered(float* C, const float* A, const float* B,
                                              int M, int N, int K, int tile_k) {
    __shared__ float As[64];
    __shared__ float Bs[64];
    int tid = threadIdx.x;
    int row = tid / N;
    int col = tid % N;

    As[row * tile_k + col] = A[row * K + col];
    Bs[row * N + col] = B[row * N + col];
    __syncthreads();

    As[row * tile_k + col] = A[row * K + tile_k + col];
    Bs[row * N + col] = B[(tile_k + row) * N + col];

    float acc = 0.0f;
    for (int k = 0; k < tile_k; k = k + 1) {
        acc = acc + As[row * tile_k + k] * Bs[k * N + col];
    }
    __syncthreads();

    for (int k = 0; k < tile_k; k = k + 1) {
        acc = acc + As[row * tile_k + k] * Bs[k * N + col];
    }

    C[row * N + col] = acc;
}
