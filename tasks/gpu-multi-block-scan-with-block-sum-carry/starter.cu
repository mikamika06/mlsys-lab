// Compute an inclusive prefix sum (scan) over 128 elements (4 blocks of
// 32) via THREE launches of this one kernel, selected by `phase`:
//   phase 0 (grid=4, block=32): per-block local inclusive scan (shfl_up
//     ladder, same as the intra-warp-scan task) written back into
//     `data` in place; lane 31 writes its block's total to
//     block_sums[blockIdx.x].
//   phase 1 (grid=1, block=4): turn block_sums from each block's TOTAL
//     into each block's CARRY -- the sum of every block strictly BEFORE
//     it (an EXCLUSIVE scan of the 4 block sums).
//   phase 2 (grid=4, block=32): add block_sums[blockIdx.x] (now the
//     carry) into every element of that block.
// See task.md for exact formulas.
__global__ void multi_block_scan(float* data, float* block_sums, int phase, int n_blocks) {
    int tid = threadIdx.x;
    // TODO: branch on `phase` (0, 1, 2) and implement each step above.
}
