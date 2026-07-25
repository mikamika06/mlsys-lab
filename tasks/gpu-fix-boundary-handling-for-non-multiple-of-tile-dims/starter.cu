// BROKEN: this ragged-tile GEMM loads every tile element unconditionally,
// with no bounds check against M, N, or K. For the last (partial) tile
// along any dimension, some threads' `grow`, `a_col`, `b_row` or `gcol`
// are past the matrix's real edge -- the load still happens, reading
// whatever real (nonzero!) data happens to sit there, and that garbage
// gets summed into the dot product for every thread in the tile, not
// just the out-of-range ones. Add the missing bounds checks (see
// task.md): zero-pad, don't read, when a tile position falls outside
// the real matrix.
__global__ void gemm_ragged_tile(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float As[16];
    __shared__ float Bs[16];
    int tid = threadIdx.x;
    int lr = tid / 4;
    int lc = tid % 4;
    int tile_row = blockIdx.x / 2;
    int tile_col = blockIdx.x % 2;
    int grow = tile_row * 4 + lr;
    int gcol = tile_col * 4 + lc;
    float acc = 0.0f;
    int num_k_tiles = (K + 3) / 4;
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int a_col = kt * 4 + lc;
        int b_row = kt * 4 + lr;
        // BUG: unconditional loads, no boundary guard.
        As[lr * 4 + lc] = A[grow * K + a_col];
        Bs[lr * 4 + lc] = B[b_row * N + gcol];
        __syncthreads();
        for (int e = 0; e < 4; e++) {
            acc += As[lr * 4 + e] * Bs[e * 4 + lc];
        }
        __syncthreads();
    }
    if (grow < M && gcol < N) {
        C[grow * N + gcol] = acc;
    }
}
