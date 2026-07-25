// Reference: shared-memory tile transpose with +1 padding.
// A single 32x32 tile, one thread per element (1024 threads, 32 warps),
// thread tid owns (row, col) = (tid / n, tid % n).
//
// Store step: tile[row*stride + col] = in[row*n + col]. A warp is 32
// consecutive tid -> fixed row, col = 0..31 -> 32 consecutive shared
// addresses -> automatically conflict-free (any padding).
//
// Load step: out[row*n + col] = tile[col*stride + row]. Same warp
// (fixed row, col = 0..31) now reads shared memory with col as the
// FAST-varying multiplier of stride. With stride = n (no padding),
// col*n mod 32 = 0 for every col (n = 32) -- every lane hits the SAME
// bank: a 32-way conflict. With stride = n + 1, col*(n+1) mod 32 =
// col mod 32, which is a bijection over the warp's 32 lanes -- every
// lane hits a DIFFERENT bank: conflict-free.
__global__ void transpose_tile(float* out, const float* in, int n) {
    __shared__ float tile[1056];  // 32 * 33, padded
    int tid = threadIdx.x;
    int row = tid / n;
    int col = tid % n;
    int stride = n + 1;
    tile[row * stride + col] = in[row * n + col];
    __syncthreads();
    out[row * n + col] = tile[col * stride + row];
}
