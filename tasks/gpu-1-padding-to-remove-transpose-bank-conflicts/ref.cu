// Reference: conflict-free 32x32 tile transpose. Shared-memory row stride
// is padded to 33 words so a transposed COLUMN read walks all 32 banks
// instead of hammering a single one.
//
// This simulator's CUDA-C frontend only carries the .x component through
// threadIdx/blockIdx/blockDim, so the kernel is launched as a single 1D
// block of 1024 (= 32*32) threads and manually decomposes threadIdx.x
// into (row, col) — exactly the (linear_tid / 32, linear_tid % 32) split
// real hardware uses internally to assign warps for a 32x32 2D block.
__global__ void transpose_tile(float* out, const float* in, int n) {
    __shared__ float tile[1056];
    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;
    tile[row * 33 + col] = in[row * n + col];
    __syncthreads();
    out[row * n + col] = tile[col * 33 + row];
}
