// Reference: inclusive prefix sum (scan) over 128 elements = 4 blocks of
// 32 (one warp each), via 3 separate launches of this SAME kernel with
// different `phase` values -- exactly how real multi-block scans work,
// since no single kernel launch can globally synchronize across blocks.
//
//   phase 0 (grid=4, block=32): each block does its own 32-lane
//   __shfl_up_sync inclusive scan (see the intra-warp-scan task) over
//   its slice of `data`, writing the LOCAL scan back in place, and lane
//   31 (the block's total) writes it to block_sums[blockIdx.x].
//
//   phase 1 (grid=1, block=4): scans the 4 block sums into an EXCLUSIVE
//   prefix -- block i's "carry" is the sum of every block BEFORE it,
//   NOT including its own total -- by running the inclusive shfl-scan
//   ladder (2 steps: delta 1, 2, since there are only 4 lanes) and then
//   subtracting each lane's own original value.
//
//   phase 2 (grid=4, block=32): every thread reads its own block's
//   carry, block_sums[blockIdx.x], and adds it to its own (already
//   locally-scanned) element.
__global__ void multi_block_scan(float* data, float* block_sums, int phase, int n_blocks) {
    if (phase == 0) {
        int tid = threadIdx.x;
        int lane = tid % 32;
        int i = blockIdx.x * blockDim.x + tid;
        float val = data[i];

        float n1 = __shfl_up_sync(0xffffffff, val, 1);
        if (lane >= 1) { val = val + n1; }
        float n2 = __shfl_up_sync(0xffffffff, val, 2);
        if (lane >= 2) { val = val + n2; }
        float n4 = __shfl_up_sync(0xffffffff, val, 4);
        if (lane >= 4) { val = val + n4; }
        float n8 = __shfl_up_sync(0xffffffff, val, 8);
        if (lane >= 8) { val = val + n8; }
        float n16 = __shfl_up_sync(0xffffffff, val, 16);
        if (lane >= 16) { val = val + n16; }

        data[i] = val;
        if (lane == 31) {
            block_sums[blockIdx.x] = val;
        }
    } else if (phase == 1) {
        int tid = threadIdx.x;
        int lane = tid % 32;
        float val = block_sums[tid];
        float orig = val;

        float n1 = __shfl_up_sync(0xffffffff, val, 1);
        if (lane >= 1) { val = val + n1; }
        float n2 = __shfl_up_sync(0xffffffff, val, 2);
        if (lane >= 2) { val = val + n2; }

        block_sums[tid] = val - orig;  // inclusive -> exclusive (the carry)
    } else {
        int tid = threadIdx.x;
        int i = blockIdx.x * blockDim.x + tid;
        float carry = block_sums[blockIdx.x];
        data[i] = data[i] + carry;
    }
}
