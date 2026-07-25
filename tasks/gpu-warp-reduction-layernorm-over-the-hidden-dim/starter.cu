// TODO: one warp per row (row = global_tid/32, lane = global_tid%32).
// Each lane sums its own D/32 strided elements (x[row*D + lane + c]
// for c = 0, 32, 64, ...) into `sum`/`sumsq`. Warp-all-reduce both via
// a butterfly XOR-shuffle: for each offset in 16, 8, 4, 2, 1 (in that
// order), read the partner lane's value into a temp variable with
// `float tmp = __shfl_xor_sync(0xffffffff, sum, offset);` then add it
// in as a SEPARATE statement (the shuffle call must be the WHOLE
// right-hand side of its own assignment). After both reductions, every
// lane holds the row's full mean/var -- normalize each of this lane's
// own elements and write them out. See ref.cu.
__global__ void warp_layernorm(const float* x, const float* gamma, const float* beta,
                                float* y, int rows, int D, float eps) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = global_tid / 32;
    int lane = global_tid % 32;
    if (row < rows) {
        // your code here
    }
}
