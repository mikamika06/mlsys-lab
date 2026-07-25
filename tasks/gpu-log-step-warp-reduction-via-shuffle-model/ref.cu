// Reference: sum-reduce each 32-lane warp using __shfl_down_sync in a
// log-step (Hillis-Steele-style) ladder -- no shared memory, no
// __syncthreads(), a warp is already synchronous. delta halves each
// step: 16, 8, 4, 2, 1. No per-lane guard is needed: for a lane whose
// (lane + delta) partner is out of range, __shfl_down_sync returns the
// lane's OWN value, so adding it in just doubles a partial that never
// gets read again -- only lane 0's accumulated value, after all 5
// steps, is ever used.
__global__ void warp_reduce_sum(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    float val = in[tid];

    float d16 = __shfl_down_sync(0xffffffff, val, 16);
    val = val + d16;
    float d8 = __shfl_down_sync(0xffffffff, val, 8);
    val = val + d8;
    float d4 = __shfl_down_sync(0xffffffff, val, 4);
    val = val + d4;
    float d2 = __shfl_down_sync(0xffffffff, val, 2);
    val = val + d2;
    float d1 = __shfl_down_sync(0xffffffff, val, 1);
    val = val + d1;

    if (lane == 0) {
        out[warp] = val;
    }
}
