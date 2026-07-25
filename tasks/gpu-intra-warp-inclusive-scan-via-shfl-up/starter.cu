// Compute an inclusive prefix sum (scan) over one warp of 32 elements:
// out[tid] = in[0] + in[1] + ... + in[tid]. Use __shfl_up_sync in a
// Hillis-Steele ladder (delta = 1, 2, 4, 8, 16) -- no shared memory, no
// __syncthreads(). __shfl_up_sync must be the WHOLE right-hand side of
// its own assignment (e.g. `float got = __shfl_up_sync(mask, val, d);`),
// so read it into its own variable first, then conditionally add it to
// `val` in a separate statement -- only for lanes where `lane - d` is a
// real lane. See task.md.
__global__ void warp_inclusive_scan(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    float val = in[tid];
    // TODO: 5-step shfl_up ladder (delta 1,2,4,8,16), each step guarded
    // by `lane >= delta`.
    out[tid] = val;
}
