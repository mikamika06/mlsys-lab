// Tiled matmul C = A @ B for n x n matrices packed into one flat buffer:
// A at `a` (offset 0), B at `b`, C at `c`. The CUDA-C frontend only
// supports 1D launches, so the 2D tiling (tiles_per_row x tiles_per_row
// blocks of 16 x 16 threads) is flattened: grid = tiles_per_row *
// tiles_per_row, block = 256, and 2D tile coordinates are recovered from
// the flat threadIdx.x / blockIdx.x.
__global__ void tiled_matmul(float* a, float* b, float* c, int n, int tiles_per_row) {
    __shared__ float sa[16 * 16];
    __shared__ float sb[16 * 16];

    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;
    int bx = blockIdx.x % tiles_per_row;
    int by = blockIdx.x / tiles_per_row;

    int row = by * 16 + ty;
    int col = bx * 16 + tx;

    float acc = 0.0;
    for (int k0 = 0; k0 < n; k0 = k0 + 16) {
        sa[ty * 16 + tx] = a[row * n + k0 + tx];
        sb[ty * 16 + tx] = b[(k0 + ty) * n + col];
        __syncthreads();
        for (int k = 0; k < 16; k = k + 1) {
            acc = acc + sa[ty * 16 + k] * sb[k * 16 + tx];
        }
        __syncthreads();
    }
    c[row * n + col] = acc;
}
