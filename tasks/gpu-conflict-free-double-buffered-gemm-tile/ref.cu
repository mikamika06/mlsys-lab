// Reference: 32x32 x 32x32 GEMM (C = A*B, K=64 split into two 32-wide
// K-tiles), launched as a single block of 1024 (=32*32) threads -- one
// thread per output element, no grid-level tiling needed since the
// whole output fits one 32x32 tile.
//
// TWO shared-memory buffers per operand, so the NEXT K-tile is loaded
// into the buffer NOT currently being read from (software-pipelined /
// double-buffered), and every tile row is padded to a 33-word stride
// (TILE+1). A whole warp here is 32 THREADS THAT SHARE THE SAME `col`
// (thread id decomposes as col = tid/32, row = tid%32) -- so at every
// compute step, the 32 lanes of a warp read As at 32 DIFFERENT, `row`-
// strided addresses. Unpadded (stride 32), `row*32 + e` collapses onto
// the SAME bank for every row (32-way conflict); padded (stride 33),
// `row*33 + e` spreads across all 32 banks.
//
// This simulator's CUDA-C frontend only carries the .x component through
// threadIdx/blockIdx/blockDim (see cuda_c.py), so the kernel manually
// decomposes the linear thread id into (row, col) exactly as real
// hardware would for a 32x32 2D block.
__global__ void gemm_tile_dbuf(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float As[2112];   // 2 buffers * 32 rows * 33 (padded stride)
    __shared__ float Bs[2112];

    int tid = threadIdx.x;
    int col = tid / 32;   // 0..31: output col -- FIXED across one warp
    int row = tid % 32;   // 0..31: output row -- varies across one warp

    // Prologue: load K-tile 0 into buffer 0.
    As[0 * 1056 + row * 33 + col] = A[row * K + 0 * 32 + col];
    Bs[0 * 1056 + row * 33 + col] = B[(0 * 32 + row) * N + col];
    __syncthreads();

    float acc = 0.0f;
    int num_k_tiles = K / 32;
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int buf = kt % 2;
        int nbuf = (kt + 1) % 2;
        if (kt + 1 < num_k_tiles) {
            // Prefetch the NEXT K-tile into the OTHER buffer while this
            // iteration is about to compute from `buf`.
            As[nbuf * 1056 + row * 33 + col] = A[row * K + (kt + 1) * 32 + col];
            Bs[nbuf * 1056 + row * 33 + col] = B[((kt + 1) * 32 + row) * N + col];
        }
        for (int e = 0; e < 32; e++) {
            acc += As[buf * 1056 + row * 33 + e] * Bs[buf * 1056 + e * 33 + col];
        }
        __syncthreads();
    }
    C[row * N + col] = acc;
}
