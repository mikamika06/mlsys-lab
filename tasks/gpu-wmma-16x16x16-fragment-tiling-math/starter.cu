// One warp (32 threads). `frag[t*8 + k]` (k in [0,8)) is lane t's k-th
// accumulator element of a 16x16 WMMA-style output tile. Scatter every
// element to its row-major position in `out` (16x16, row-major):
//   row = (t / 4) + 8 * (k / 4)
//   col = (t % 4) * 4 + (k % 4)
__global__ void wmma_store_c(float* out, const float* frag) {
    int t = threadIdx.x;
    // your code here
}
