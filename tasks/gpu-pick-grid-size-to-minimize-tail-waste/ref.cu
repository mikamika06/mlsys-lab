// A grid-stride kernel can be launched with FEWER blocks than
// ceil(N/block_size) needs -- each block just loops to cover more
// elements. The GPU can only run `max_concurrent` blocks at once; any
// grid runs in ceil(num_blocks/max_concurrent) back-to-back WAVES, and
// if num_blocks isn't a multiple of max_concurrent, the LAST wave is
// only partially full -- SM slots sit idle for that whole wave.
//
// Choose num_blocks = the LARGEST multiple of max_concurrent that is
// <= the natural block count (ceil(N/block_size)) -- every wave is
// then completely full, no tail waste, at the cost of each block
// covering slightly more elements via its grid-stride loop. If the
// natural block count doesn't even reach one full wave
// (total_blocks <= max_concurrent), there's only one wave regardless,
// so just use the natural count.
__global__ void optimal_grid_blocks(float* out, int idx, int N, int block_size, int max_concurrent) {
    int total_blocks = (N + block_size - 1) / block_size;
    int best = total_blocks;
    if (total_blocks > max_concurrent) {
        best = (total_blocks / max_concurrent) * max_concurrent;
    }
    out[idx] = best;
}
