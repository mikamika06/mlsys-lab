// Reference: double-buffered 2-tile matmul. Tile 0 loads into As0/Bs0 and
// is used for the first partial sum; tile 1 is PREFETCHED into the OTHER
// pair of buffers (As1/Bs1) placed right after the first sync, with no
// barrier separating the prefetch from the tile-0 compute loop that
// follows it -- safe, because they touch different shared arrays. Only
// after both are done does a second sync guard the switch to tile 1.
__global__ void tiled_matmul_double_buffered(float* C, const float* A, const float* B,
                                              int M, int N, int K, int tile_k) {
    __shared__ float As0[64];
    __shared__ float Bs0[64];
    __shared__ float As1[64];
    __shared__ float Bs1[64];
    int tid = threadIdx.x;
    int row = tid / N;
    int col = tid % N;

    As0[row * tile_k + col] = A[row * K + col];
    Bs0[row * N + col] = B[row * N + col];
    __syncthreads();

    As1[row * tile_k + col] = A[row * K + tile_k + col];
    Bs1[row * N + col] = B[(tile_k + row) * N + col];

    float acc = 0.0f;
    for (int k = 0; k < tile_k; k = k + 1) {
        acc = acc + As0[row * tile_k + k] * Bs0[k * N + col];
    }
    __syncthreads();

    for (int k = 0; k < tile_k; k = k + 1) {
        acc = acc + As1[row * tile_k + k] * Bs1[k * N + col];
    }

    C[row * N + col] = acc;
}
