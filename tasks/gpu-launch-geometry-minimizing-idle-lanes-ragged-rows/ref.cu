// Ragged rows, COMPACTED (not padded): row r's lens[r] real elements are
// concatenated back-to-back in `data`, no gaps -- so the launch only
// ever needs `total = sum(lens)` threads, never one per padded slot.
// Each flat thread id `tid` must find which row it belongs to via the
// prefix-sum `offsets` array (offsets[r] = sum of lens[0..r), so row r
// owns flat positions [offsets[r], offsets[r+1])): scan every offset
// and remember the last one that's still `<= tid` -- that's the row.
__global__ void ragged_process(float* out, const float* data, const int* offsets,
                                const float* row_scale, int R, int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total) {
        int row = 0;
        int r = 0;
        while (r < R) {
            if (offsets[r] <= tid) {
                row = r;
            }
            r = r + 1;
        }
        out[tid] = data[tid] + row_scale[row];
    }
}
