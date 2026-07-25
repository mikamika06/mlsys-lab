// Reference: inclusive prefix-sum (scan) over one warp of 32 lanes,
// using __shfl_up_sync -- no shared memory, no __syncthreads(), a warp
// is already synchronous. Classic Hillis-Steele ladder: at step `delta`
// (1, 2, 4, 8, 16), lane `lane` receives lane `lane - delta`'s CURRENT
// running value and adds it in -- but only if `lane - delta` is a real
// lane (lane >= delta). For lane < delta, __shfl_up_sync's source is
// out of range and the simulator (like real hardware) returns the
// lane's OWN value back -- adding that in unconditionally would double
// it, which is why every step needs the guard.
__global__ void warp_inclusive_scan(float* out, const float* in, int n) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    float val = in[tid];

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

    out[tid] = val;
}
