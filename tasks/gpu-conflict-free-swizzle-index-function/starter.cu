// TODO: store the N x N tile into shared memory through a
// conflict-free swizzle, then read one column back through the same
// swizzle. Use phys(row, col) = row*32 + (row+col)%32 -- diagonal
// shift by `row` mod 32 -- for both the write loop and the final read.
// See task.md for why this makes every 32-lane access hit 32 distinct
// banks instead of 1.
__global__ void swizzle_roundtrip(const float* in, float* out, int target_col) {
    __shared__ float tile[1024];
    int row = threadIdx.x;
    // your code here
}
