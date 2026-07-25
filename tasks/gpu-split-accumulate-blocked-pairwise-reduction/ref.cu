// Reference: split-accumulate (blocked) reduction. Sum values[0..n) in
// groups of `block_size`: each group gets its OWN fresh accumulator
// (block_sum, reset to 0 for every new block), and only the finished
// block totals get folded into the running grand total. A block whose
// own elements happen to cancel out (or stay small) never lets its
// internal magnitude swings leak into the running total of any OTHER
// block.
__global__ void split_accumulate(float* out, const float* values, int n, int block_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float total = 0.0f;
        int i = 0;
        while (i < n) {
            float block_sum = 0.0f;
            for (int j = 0; j < block_size; j++) {
                block_sum = block_sum + values[i + j];
            }
            total = total + block_sum;
            i = i + block_size;
        }
        out[0] = total;
    }
}
