// BROKEN: every one of the 256 threads is supposed to add 1.0 to a
// single shared counter, so out[0] should end up 256.0. Instead, EVERY
// thread reads the counter's CURRENT value, THEN (only after a barrier
// forces every thread to have already done its read) computes and
// writes back its own "current + 1" -- so all 256 threads read the same
// stale 0.0, all compute 1.0, and whichever one's write lands last is
// the only increment that survives. This simulator (like real hardware)
// has no atomic add; the fix is to restructure the algorithm so no two
// threads ever read-modify-write the same address, not to try to
// synchronize the race away.
__global__ void race_free_count(float* out, int n) {
    __shared__ float counter[1];
    int tid = threadIdx.x;
    if (tid == 0) { counter[0] = 0.0f; }
    __syncthreads();

    float v = counter[0];
    __syncthreads();
    // BUG: every thread computed its "+1" from the SAME pre-barrier
    // read of counter[0] -- 256 threads all overwrite counter[0] with
    // the same 1.0, not 256 distinct increments.
    v = v + 1.0f;
    counter[0] = v;
    __syncthreads();

    if (tid == 0) {
        out[0] = counter[0];
    }
}
