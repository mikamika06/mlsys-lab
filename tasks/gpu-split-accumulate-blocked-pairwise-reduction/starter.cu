// Sum values[0..n) using a split-accumulate (blocked) reduction: process
// elements in consecutive groups of `block_size`, accumulating each
// group into its OWN fresh local accumulator (reset to 0 at the start of
// every group), then add each finished group's total into a running
// grand total.
__global__ void split_accumulate(float* out, const float* values, int n, int block_size) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // your code here
    }
}
