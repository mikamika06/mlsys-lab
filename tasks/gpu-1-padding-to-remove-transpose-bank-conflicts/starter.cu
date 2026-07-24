// Starter: correct transpose, but the shared-memory tile is NOT padded
// (row stride 32, matching the bank count exactly). Every column read
// after the barrier lands in the same bank for all 32 lanes of a warp —
// a 32-way bank conflict on every step.
__global__ void transpose_tile(float* out, const float* in, int n) {
    __shared__ float tile[1024];
    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;
    tile[row * 32 + col] = in[row * n + col];
    __syncthreads();
    out[row * n + col] = tile[col * 32 + row];
}
