// Reference: reduce 32 partial sums (one per lane, already the output of
// some earlier reduction stage) down to a single total using ONLY warp
// shuffles -- no __shared__ array, no __syncthreads(). A warp is already
// synchronous, so once a reduction is down to 32 live values there is no
// reason left to pay for shared-memory traffic and barriers at all.
__global__ void warp_final_reduce(float* out, const float* partial, int n) {
    int tid = threadIdx.x;
    float val = partial[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (tid == 0) {
        out[0] = val;
    }
}
