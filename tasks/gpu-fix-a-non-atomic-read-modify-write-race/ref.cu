// Reference: count 256 threads' contributions (1.0 each) WITHOUT any
// atomic primitive, by avoiding the race entirely instead of trying to
// synchronize around it: every thread writes its own contribution into
// its OWN shared-memory slot (sdata[tid] = 1.0f -- no two threads ever
// target the same address), then a standard barrier-synchronized tree
// reduction combines all 256 slots into one total. No two threads ever
// read-modify-write the SAME address at the same step, so there is
// nothing left to race on.
__global__ void race_free_count(float* out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = 1.0f;
    __syncthreads();
    for (int stride = 128; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] = sdata[tid] + sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = sdata[0];
    }
}
