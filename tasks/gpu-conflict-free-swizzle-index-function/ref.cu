// Reference: diagonal swizzle phys(row,col) = row*32 + (row+col)%32.
// Store an N x N tile into shared memory through the swizzle, then read
// one column back through the same swizzle. Both the write sweep (fixed
// col, varying row across the warp) and the read (fixed target column,
// one word per thread) are the classic bank-conflict-prone patterns for
// a plain row-major layout -- the swizzle fixes both simultaneously.
__global__ void swizzle_roundtrip(const float* in, float* out, int target_col) {
    __shared__ float tile[1024];
    int row = threadIdx.x;

    int col = 0;
    while (col < 32) {
        int phys = row * 32 + (row + col) % 32;
        tile[phys] = in[row * 32 + col];
        col = col + 1;
    }
    __syncthreads();

    int rphys = row * 32 + (row + target_col) % 32;
    out[row] = tile[rphys];
}
