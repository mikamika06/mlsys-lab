// Tiled matmul: C = A * B for M x K times K x N, using 16x16 tiles held
// in shared memory. Each block computes one 16x16 tile of C. For each of
// the K/16 tile-steps along the K dimension, the block cooperatively
// loads one 16x16 tile of A and one of B into shared memory (each global
// element loaded ONCE per block, by exactly one thread), then every
// thread in the block reuses that same tile out of shared memory 16
// times (once per k within the tile) to accumulate its own C entry --
// instead of every thread re-reading the same A/B elements from global
// memory itself.
//
// This simulator's CUDA-C frontend is 1D-only (grid/block are ints, and
// threadIdx.y/blockIdx.y always read as 0) -- so the 2D tile/thread
// coordinates are derived by hand from the linear threadIdx.x/blockIdx.x.
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

    float acc = 0.0f;
    int numTiles = K / 16;
    int kt = 0;
    while (kt < numTiles) {
        As[ty * 16 + tx] = A[row * K + kt * 16 + tx];
        Bs[ty * 16 + tx] = B[(kt * 16 + ty) * N + col];
        __syncthreads();

        int k = 0;
        while (k < 16) {
            acc += As[ty * 16 + k] * Bs[k * 16 + tx];
            k = k + 1;
        }
        __syncthreads();
        kt = kt + 1;
    }

    C[row * N + col] = acc;
}
