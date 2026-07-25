// Same shared-memory tile transpose, but WITHOUT padding: tile[32*32],
// stride = n. This computes the exact right VALUES (still a correct
// transpose) but the load step tile[col*n + row] has every lane of a
// warp (fixed row, col = 0..31) landing on the SAME bank -- a 32-way
// shared-memory bank conflict. Add +1 padding to the tile's row stride
// to fix it: use a 33-wide row (tile[1056]) and stride = n + 1 for both
// the store and the load.
__global__ void transpose_tile(float* out, const float* in, int n) {
    __shared__ float tile[1024];  // 32 * 32, NOT padded -- the bug
    int tid = threadIdx.x;
    int row = tid / n;
    int col = tid % n;
    tile[row * n + col] = in[row * n + col];
    __syncthreads();
    out[row * n + col] = tile[col * n + row];
}
