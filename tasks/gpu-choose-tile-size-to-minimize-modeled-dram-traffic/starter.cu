// Tiled matmul: C = A @ B for n x n matrices packed into one flat buffer,
// A at `a` (offset 0), B at `b`, C at `c`. Launch is 1D:
// grid = tiles_per_row * tiles_per_row blocks of 256 threads each. Recover
// 2D tile coordinates from the flat threadIdx.x / blockIdx.x:
//   tx = threadIdx.x % 16;   ty = threadIdx.x / 16;
//   bx = blockIdx.x % tiles_per_row;   by = blockIdx.x / tiles_per_row;
//   row = by * 16 + ty;   col = bx * 16 + tx;
//
// TODO: for k0 = 0, 16, 32, ... < n: stage a[row][k0+tx] and b[k0+ty][col]
// into two __shared__ 16*16 tiles, __syncthreads(), accumulate 16 products
// from shared memory, __syncthreads(), then write the accumulator to
// c[row][col].
__global__ void tiled_matmul(float* a, float* b, float* c, int n, int tiles_per_row) {
    // your code here
}
