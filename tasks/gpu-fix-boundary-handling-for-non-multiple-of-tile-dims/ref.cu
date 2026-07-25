// Reference: 6x6 x 6x6 GEMM (C = A*B), tiled 4x4 -- M, N, K = 6 are NOT
// multiples of the 4x4 tile, so every tile along every edge is "ragged":
// the last row-tile only has 2 valid rows (of 4), the last col-tile only
// 2 valid cols, the last K-tile only 2 valid K values. Launched as a 2x2
// grid (ceil(6/4)=2 per dim) of 16-thread (4x4) blocks; each thread
// loads its tile element ONLY if both its row/col AND its K-position are
// in range, else stores 0.0 -- a ragged tile behaves exactly like a full
// tile whose out-of-range corner is zero-padded, so the dot product is
// unaffected by whatever real data happens to live just past the
// matrix's edge in memory.
//
// This simulator's CUDA-C frontend only carries the .x component through
// threadIdx/blockIdx/blockDim (see cuda_c.py), so the 2x2 tile grid is
// launched as 4 flattened 1D blocks, manually decomposed into
// (tile_row, tile_col), exactly like the thread id is decomposed into
// (local row, local col).
__global__ void gemm_ragged_tile(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float As[16];   // 4x4 tile
    __shared__ float Bs[16];

    int tid = threadIdx.x;
    int lr = tid / 4;
    int lc = tid % 4;
    int tile_row = blockIdx.x / 2;
    int tile_col = blockIdx.x % 2;
    int grow = tile_row * 4 + lr;
    int gcol = tile_col * 4 + lc;

    float acc = 0.0f;
    int num_k_tiles = (K + 3) / 4;  // ceil(K/4)
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int a_col = kt * 4 + lc;
        int b_row = kt * 4 + lr;

        if (grow < M && a_col < K) {
            As[lr * 4 + lc] = A[grow * K + a_col];
        } else {
            As[lr * 4 + lc] = 0.0f;
        }
        if (b_row < K && gcol < N) {
            Bs[lr * 4 + lc] = B[b_row * N + gcol];
        } else {
            Bs[lr * 4 + lc] = 0.0f;
        }
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
