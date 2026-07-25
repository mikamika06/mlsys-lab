// Reference: store a WMMA-style 16x16 accumulator fragment out to its
// row-major C-matrix layout. `frag` holds the 256 accumulator values in
// THREAD-MAJOR fragment order (frag[t*8 + k] is lane t's k-th of 8
// elements, k in [0,8)) -- the layout a real 16x16x16 tensor-core MMA
// leaves its results in, split 8 elements per lane across one warp.
//
// This task's fragment-to-element mapping (deliberately simplified, not
// meant to match any specific real hardware's exact PTX layout):
//   row = (t / 4) + 8 * (k / 4)
//   col = (t % 4) * 4 + (k % 4)
// Lane t owns two 4-wide column runs, one in row (t/4) and one 8 rows
// below it in row (t/4 + 8) -- every (row, col) in the 16x16 tile is
// covered by exactly one (t, k) pair.
__global__ void wmma_store_c(float* out, const float* frag) {
    int t = threadIdx.x;
    for (int k = 0; k < 8; k++) {
        int row = (t / 4) + 8 * (k / 4);
        int col = (t % 4) * 4 + (k % 4);
        out[row * 16 + col] = frag[t * 8 + k];
    }
}
